from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


def _infer_project_root() -> Path:
    # config.py -> src -> slot_ir -> repository root (contains data/)
    return Path(__file__).resolve().parents[2]


PROJECT_ROOT = _infer_project_root()
SLOT_UNMIXER_ROOT = Path(__file__).resolve().parents[1]  # the slot_ir/ directory
DATA_ROOT = Path(os.environ.get("INFRARED_DATA_ROOT", PROJECT_ROOT / "data"))


@dataclass
class DataConfig:
    processed_dir: Path = DATA_ROOT / "processed"
    processed_unmix_dir: Path = DATA_ROOT / "processed_unmix"
    mixture_dataset: str = "mix_ml_instances.h5"
    drug_mixture_dataset: str = "mix_drugs_ml_instances.h5"
    batch_size: int = 64
    num_workers: int = 4
    pin_memory: bool = True
    use_global_intensity_scale: bool = True
    global_intensity_scale: float = 1.0  # Target max after scaling (for mixtures)
    n_holdout_molecules: int = 0  # Number of molecules to hold out from Stage 2 mixture training
    # Seed for choosing the held-out molecules (None = use the model seed). run_holdout.py sets 42.
    holdout_seed: Optional[int] = None
    include_drug_data: bool = False  # Include drug mixtures in Stage 2 training


@dataclass
class ModelConfig:
    spectrum_dim: int = 1250
    d_model: int = 512
    n_heads: int = 8
    n_encoder_layers: int = 6
    n_cross_layers: int = 4
    n_tokens: int = 256
    token_window_sizes: List[int] = field(default_factory=lambda: [10, 25, 50])
    max_components: int = 2  # Default to binary mixtures
    functional_groups: int = 10
    dropout: float = 0.5
    confidence_temperature: float = 2.0


@dataclass
class OptimizationConfig:
    lr: float = 5e-5
    warmup_steps: int = 500
    weight_decay: float = 1e-5
    max_grad_norm_stage_pretrain: float = 0.5
    max_grad_norm_self_train: float = 0.25
    stage_loss_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "translation": 1.0,
            "component": 1.0,
        }
    )


@dataclass
class AugmentationConfig:
    enable: bool = False
    # Compositional augmentation: generate synthetic mixtures from random molecule combinations
    num_synthetic_samples: int = 0  # Number of synthetic mixtures to add to training set (ignored if generate_all_combinations=True)
    source: str = "train"  # Molecule pool: "train", "drug", "drug_train", "stratified"
    num_drug_train_samples: int = 0  # For stratified: additional drug+train pairs
    include_s1_drugs: bool = False  # If True, include S1 drug molecules (in training DB) in the drug pool
    balance_rare_molecules: bool = False  # Weight training molecule sampling by inverse frequency

    # Sampling strategy for synthetic mixtures
    sampling_strategy: str = "random"  # Options: random, hard_spectral, hard_functional, diverse
    hard_negative_temperature: float = 0.5  # For hard negative sampling: lower = harder (more similar)
    generate_all_combinations: bool = False  # If True, generate ALL possible K-combinations (can be very large)

    # Variable concentration: use random weights instead of equal proportions
    variable_concentration: bool = False
    concentration_min: float = 0.1  # Minimum weight for any component (e.g., 0.1 = 10%)

    # Spectral augmentation: apply shift_windows to each component before mixing
    # shift_windows parameters control the augmentation strength
    shift_percent: float = 0.2  # Fraction of spectrum points to shift (0.0-1.0)
    shift_window_width: int = 30  # Width of each shifted window (in wavenumbers)
    shift_range: float = 5.0  # Gaussian std of shift amount (in wavenumbers)


@dataclass
class StageConfig:
    stage: int = 1
    freeze_upper_levels_steps: int = 200
    teacher_forcing_start: float = 1.0
    teacher_forcing_end: float = 0.2
    teacher_forcing_decay_steps: int = 5000
    checkpoint_path: Optional[Path] = None
    max_epochs: Optional[int] = None
    early_stopping_patience: Optional[int] = None  # None = disabled, int = epochs to wait
    early_stopping_min_delta_loss: float = 0.0001  # Minimum improvement in loss
    early_stopping_min_delta_metric: float = 0.001  # Minimum improvement in metric (e.g., cosine)
    # Stage 1 objective: "translation" (gas->liquid, default), "autoencoder" (liquid->liquid),
    # "gas_autoencoder" (gas->gas), or "shuffled" (gas_i->liquid_j control)
    stage1_mode: str = "translation"
    # Stage 2: initialize the slot projection and prediction head from the model seed
    # instead of copying them from the Stage 1 checkpoint
    reinit_heads: bool = False


@dataclass
class VisualizationConfig:
    save_dir: Path = SLOT_UNMIXER_ROOT / "results"
    max_samples: int = 16
    dpi: int = 180
    # Colors matching ir_interpreter palette
    color_mixture: str = "#b8336a"  # ground_truth (pink-red)
    color_component_a: str = "#456991"  # gas (deep blue)
    color_component_b: str = "#c2768b"  # liquid (pink/magenta)
    color_component_c: str = "#6a994e"  # third component (green)
    color_gas: str = "#456991"  # gas (deep blue)
    color_liquid_true: str = "#b8336a"  # ground_truth (pink-red)
    color_liquid_pred: str = "#f7797d"  # predicted (coral)
    attention_cmap: str = "viridis"
    interpreter_palette: Optional[Dict[str, str]] = None
    
    def __post_init__(self):
        """Initialize interpreter_palette if not provided."""
        if self.interpreter_palette is None:
            self.interpreter_palette = {
                'gas': self.color_gas,
                'predicted': self.color_liquid_pred,
                'ground_truth': self.color_liquid_true,
                'difference': '#C73E1D'  # Red for difference
            }


@dataclass
class TrainingConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimizationConfig = field(default_factory=OptimizationConfig)
    stage: StageConfig = field(default_factory=StageConfig)
    viz: VisualizationConfig = field(default_factory=VisualizationConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    seed: int = 42
    device: str = "cuda"
    log_every: int = 50
    val_every: int = 500
    save_every: int = 1000
    results_dir: Path = SLOT_UNMIXER_ROOT / "results"


def load_config(**overrides) -> TrainingConfig:
    cfg = TrainingConfig()
    training_overrides = overrides.pop("training", None)
    for key, value in overrides.items():
        if not hasattr(cfg, key):
            raise AttributeError(f"Unknown config section: {key}")
        section = getattr(cfg, key)
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if not hasattr(section, sub_key):
                    print(f"[Config] Ignoring unknown field '{sub_key}' in section '{key}'")
                    continue
                # Convert string paths to Path objects
                if sub_key.endswith("_dir") or sub_key == "checkpoint_path" or sub_key == "save_dir":
                    sub_value = Path(sub_value) if isinstance(sub_value, str) else sub_value
                setattr(section, sub_key, sub_value)
        else:
            # Convert string paths to Path objects for top-level fields
            if key == "results_dir" or key.endswith("_dir"):
                value = Path(value) if isinstance(value, str) else value
            setattr(cfg, key, value)

    if training_overrides:
        if "batch_size" in training_overrides:
            cfg.data.batch_size = training_overrides["batch_size"]
        if "num_workers" in training_overrides:
            cfg.data.num_workers = training_overrides["num_workers"]
        if "device" in training_overrides:
            cfg.device = training_overrides["device"]
        if "epochs" in training_overrides:
            cfg.stage.max_epochs = training_overrides["epochs"]
    
    # Sync visualization save_dir with results_dir
    cfg.viz.save_dir = cfg.results_dir
    
    return cfg

