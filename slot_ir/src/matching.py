"""Component IR to SMILES matching using similarity search."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from .config import DataConfig, DATA_ROOT


class LiquidIRDatabase:
    """Database of liquid IR spectra with SMILES for similarity matching."""

    def __init__(
        self,
        cfg: DataConfig,
        apply_normalization: bool = True,
        device: Optional[Union[str, torch.device]] = None,
        database_file: Optional[str] = None,
    ):
        """Initialize the database from processed data."""
        import h5py

        # Determine which database to use
        if database_file is None:
            database_file = "combined_molecule_database.h5"
        db_path = cfg.processed_unmix_dir / database_file
        if not db_path.exists():
            # Fall back to processed/liq_spectra.pt if the HDF5 file is missing
            processed = cfg.processed_dir
            metadata_path = processed / "metadata.json"
            liq_spectra = torch.load(processed / "liq_spectra.pt")
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            smiles_list = metadata.get('smiles_list', [])
        else:
            # Load from HDF5 (same source as training data)
            with h5py.File(db_path, 'r') as f:
                liq_spectra = torch.from_numpy(f["liq_spectra"][:]).float()
                smiles_list = [s.decode('utf-8') if isinstance(s, bytes) else s for s in f["smiles"][:]]
        
        # Apply normalization if needed (same as training)
        if apply_normalization:
            from .data import get_global_scale_factor
            scale_factor = get_global_scale_factor(cfg)
            liq_spectra = (liq_spectra * scale_factor).clamp(max=1.0)
        
        # Store data
        # Set device preference
        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        
        self.spectra = liq_spectra.to(self.device)  # [N, L]
        self.smiles = smiles_list  # List of N SMILES strings
        self.n_samples = len(smiles_list)
        
        # Normalize spectra for efficient cosine similarity computation
        self.spectra_norm = F.normalize(self.spectra, p=2, dim=1)  # [N, L] # 8,550
        
        print(f"[Database] Loaded {self.n_samples} liquid IR spectra with SMILES")
    
    def find_best_match(
        self, 
        query_spectrum: torch.Tensor, 
        top_k: int = 50,
        exclude_smiles: Optional[List[str]] = None
    ) -> List[Tuple[str, float, int]]:
        """Find best matching SMILES for a query spectrum using cosine similarity."""
        # Normalize query
        if query_spectrum.dim() == 1:
            query_spectrum = query_spectrum.unsqueeze(0)
        
        query_norm = F.normalize(query_spectrum, p=2, dim=1).to(self.device)  # [1, L]
        
        # Compute cosine similarity: [1, L] @ [N, L].T = [1, N]
        similarities = torch.mm(query_norm, self.spectra_norm.T)  # [1, N]
        similarities = similarities.squeeze(0)  # [N]
        
        # Exclude specific SMILES if requested
        if exclude_smiles:
            exclude_indices = [
                i for i, smiles in enumerate(self.smiles) 
                if smiles in exclude_smiles
            ]
            if exclude_indices:
                similarities[exclude_indices] = -1.0  # Set to minimum similarity
        
        # Get top-k matches
        top_k = min(top_k, self.n_samples)
        top_similarities, top_indices = torch.topk(similarities, k=top_k, dim=0)
        
        results = []
        for sim, idx in zip(top_similarities.cpu().numpy(), top_indices.cpu().numpy()):
            smiles = self.smiles[idx]
            results.append((smiles, float(sim), int(idx)))
        
        return results
    
    def get_spectrum_by_smiles(self, smiles: str) -> Optional[torch.Tensor]:
        """Get liquid IR spectrum for a given SMILES string."""
        try:
            idx = self.smiles.index(smiles)
            return self.spectra[idx]
        except ValueError:
            return None


def _count_hits_on_pairs(
    pair: Tuple[List[int], List[int]],
    gt_smiles_list: List[str],
    top_k_candidates: List[List[Tuple[str, float]]],
) -> Dict[str, int]:
    """Strict R@K: count hits only on the given (predicted, target) pairs."""
    pred_idx, tgt_idx = pair
    out = {}
    for name, cutoff in (("n_top1", 1), ("n_top5", 5), ("n_top10", 10), ("n_top50", 50)):
        out[name] = sum(
            int(gt_smiles_list[t] in {s for s, _ in top_k_candidates[p][:cutoff]})
            for p, t in zip(pred_idx, tgt_idx)
        )
    return out


def _compute_order_invariant_accuracy(
    matched_smiles_list: List[str],
    gt_smiles_list: List[str],
    top_k_candidates: List[List[Tuple[str, float]]],
) -> Dict[str, int]:
    """Compute order-invariant matching accuracy for a single sample at
    multiple cutoffs (top-1, top-5, top-10).
    """
    from scipy.optimize import linear_sum_assignment

    n_pred = len(matched_smiles_list)
    n_gt = len(gt_smiles_list)

    # Top-1: cost=0 if match, cost=1 otherwise
    cost_top1 = np.ones((n_pred, n_gt), dtype=np.float32)
    for i, pred_smiles in enumerate(matched_smiles_list):
        if pred_smiles is None:
            continue
        for j, gt_smiles in enumerate(gt_smiles_list):
            if pred_smiles == gt_smiles:
                cost_top1[i, j] = 0.0
    row1, col1 = linear_sum_assignment(cost_top1)
    n_top1 = int((cost_top1[row1, col1] == 0).sum())

    # Top-5: cost=0 if GT in first 5 candidates
    cost_top5 = np.ones((n_pred, n_gt), dtype=np.float32)
    for i, candidates in enumerate(top_k_candidates):
        candidate_smiles = {s for s, _ in candidates[:5]}
        for j, gt_smiles in enumerate(gt_smiles_list):
            if gt_smiles in candidate_smiles:
                cost_top5[i, j] = 0.0
    row5, col5 = linear_sum_assignment(cost_top5)
    n_top5 = int((cost_top5[row5, col5] == 0).sum())

    # Top-10: cost=0 if GT in first 10 candidates
    cost_top10 = np.ones((n_pred, n_gt), dtype=np.float32)
    for i, candidates in enumerate(top_k_candidates):
        candidate_smiles = {s for s, _ in candidates[:10]}
        for j, gt_smiles in enumerate(gt_smiles_list):
            if gt_smiles in candidate_smiles:
                cost_top10[i, j] = 0.0
    row10, col10 = linear_sum_assignment(cost_top10)
    n_top10 = int((cost_top10[row10, col10] == 0).sum())

    # Top-50: cost=0 if GT in first 50 candidates
    cost_top50 = np.ones((n_pred, n_gt), dtype=np.float32)
    for i, candidates in enumerate(top_k_candidates):
        candidate_smiles = {s for s, _ in candidates[:50]}
        for j, gt_smiles in enumerate(gt_smiles_list):
            if gt_smiles in candidate_smiles:
                cost_top50[i, j] = 0.0
    row50, col50 = linear_sum_assignment(cost_top50)
    n_top50 = int((cost_top50[row50, col50] == 0).sum())

    return {"n_top1": n_top1, "n_top5": n_top5, "n_top10": n_top10, "n_top50": n_top50}


def match_components_to_smiles(
    predicted_components: torch.Tensor,
    database: LiquidIRDatabase,
    ground_truth_smiles: Optional[List[List[str]]] = None,
    top_k: int = 50,
    pairs: Optional[List[Tuple[List[int], List[int]]]] = None,
) -> Dict:
    """Match predicted component IRs to SMILES using similarity search."""
    batch_size, n_components, _ = predicted_components.shape

    matched_smiles = []
    similarities = []
    top_k_matches = []

    # Per-sample accuracy tracking
    sample_top1_correct = []
    sample_top5_correct = []
    sample_top10_correct = []
    sample_top50_correct = []

    for b in range(batch_size):
        batch_matched = []
        batch_similarities = []
        batch_top_k = []

        for k in range(n_components):
            query = predicted_components[b, k]  # [L]
            matches = database.find_best_match(query, top_k=top_k)

            if matches:
                matched_smiles_k, similarity_k, _ = matches[0]
                batch_matched.append(matched_smiles_k)
                batch_similarities.append(similarity_k)
                batch_top_k.append([(s, sim) for s, sim, _ in matches])
            else:
                batch_matched.append(None)
                batch_similarities.append(0.0)
                batch_top_k.append([])

        matched_smiles.append(batch_matched)
        similarities.append(batch_similarities)
        top_k_matches.append(batch_top_k)

        # Compute order-invariant accuracy at multiple cutoffs
        if ground_truth_smiles:
            gt_smiles_batch = ground_truth_smiles[b]
            if pairs is not None:
                acc = _count_hits_on_pairs(pairs[b], gt_smiles_batch, batch_top_k)
            else:
                acc = _compute_order_invariant_accuracy(
                    batch_matched, gt_smiles_batch, batch_top_k
                )
            sample_top1_correct.append(acc["n_top1"])
            sample_top5_correct.append(acc["n_top5"])
            sample_top10_correct.append(acc["n_top10"])
            sample_top50_correct.append(acc["n_top50"])

    results = {
        'matched_smiles': matched_smiles,
        'similarities': similarities,
        'top_k_matches': top_k_matches,
    }

    if ground_truth_smiles:
        total_components = batch_size * n_components
        results['top1_accuracy'] = float(np.sum(sample_top1_correct)) / total_components
        results['top5_accuracy'] = float(np.sum(sample_top5_correct)) / total_components
        results['top10_accuracy'] = float(np.sum(sample_top10_correct)) / total_components
        results['top50_accuracy'] = float(np.sum(sample_top50_correct)) / total_components
        results['per_sample_top1_correct'] = sample_top1_correct
        results['per_sample_top5_correct'] = sample_top5_correct
        results['per_sample_top10_correct'] = sample_top10_correct
        results['per_sample_top50_correct'] = sample_top50_correct

    return results


def align_components_to_ground_truth(
    predicted: torch.Tensor,
    ground_truth: torch.Tensor,
) -> Tuple[torch.Tensor, List[int], List[float]]:
    """Reorder predicted components so that they best align with the ground truth components."""
    if predicted.size(0) != ground_truth.size(0):
        raise ValueError("Predicted and ground truth component counts must match.")
    k = predicted.size(0)
    pred_norm = F.normalize(predicted, p=2, dim=1)
    gt_norm = F.normalize(ground_truth, p=2, dim=1)
    similarity_matrix = torch.mm(pred_norm, gt_norm.transpose(0, 1))  # [K, K]

    assignments = {}
    used_pred = set()
    used_gt = set()
    flat_indices = torch.argsort(similarity_matrix.reshape(-1), descending=True)

    for flat_idx in flat_indices:
        pred_idx = int(flat_idx) // k
        gt_idx = int(flat_idx) % k
        if pred_idx in used_pred or gt_idx in used_gt:
            continue
        assignments[gt_idx] = pred_idx
        used_pred.add(pred_idx)
        used_gt.add(gt_idx)
        if len(assignments) == k:
            break

    if len(assignments) < k:
        remaining_gt = [gt for gt in range(k) if gt not in assignments]
        remaining_pred = [pred for pred in range(k) if pred not in used_pred]
        for gt_idx, pred_idx in zip(remaining_gt, remaining_pred):
            assignments[gt_idx] = pred_idx

    ordered_pred_indices = [assignments[gt_idx] for gt_idx in range(k)]
    aligned_predicted = predicted[ordered_pred_indices]
    alignment_sims = [
        float(similarity_matrix[gt_idx, assignments[gt_idx]].item()) for gt_idx in range(k)
    ]
    return aligned_predicted, ordered_pred_indices, alignment_sims


class RefinementMatcher:
    """Hybrid ranking with residual refinement for binary mixtures."""

    def __init__(
        self,
        database: LiquidIRDatabase,
        top_n_per_slot: int = 5,
        top_n_pairs: int = 5,
        recon_weight: float = 0.75,
        high_confidence_threshold: float = 0.98,
    ):
        self.database = database
        self.top_n_per_slot = top_n_per_slot
        self.top_n_pairs = top_n_pairs
        self.recon_weight = recon_weight
        self.high_confidence_threshold = high_confidence_threshold

        self._db_spectra_np = np.clip(
            database.spectra.cpu().numpy().astype(np.float64), 0, 1,
        )
        print(f"[RefinementMatcher] Initialized: {database.n_samples} molecules, "
              f"top_n_per_slot={top_n_per_slot}, top_n_pairs={top_n_pairs}")

    def match_components_joint(
        self,
        predicted_components: torch.Tensor,
        mixture: torch.Tensor,
        ground_truth_smiles: Optional[List[str]] = None,
        top_k: int = 5,
    ) -> Dict:
        """Match binary mixture components with SELECTIVE residual refinement."""
        K = predicted_components.shape[0]
        if K != 2:
            raise ValueError(f"Only K=2 supported, got K={K}")

        mixture_np = mixture.cpu().numpy().astype(np.float64)
        smiles_list = self.database.smiles

        # Stage A: Per-component cosine retrieval
        per_slot_matches = []
        for k in range(K):
            matches = self.database.find_best_match(
                predicted_components[k],
                top_k=self.top_n_per_slot
            )
            per_slot_matches.append(matches)

        # Extract top-1 scores and indices
        top1_scores = [matches[0][1] if matches else 0.0 for matches in per_slot_matches]
        top1_indices = [matches[0][2] if matches else -1 for matches in per_slot_matches]

        # Determine which components are fixed (high confidence >= 0.98)
        fixed_components = [
            score >= self.high_confidence_threshold
            for score in top1_scores
        ]

        # STRATEGY 1: BYPASS_BOTH
        if all(fixed_components):
            joint_top_k = self._build_cosine_pairs(per_slot_matches, top_k)
            result = self._build_result(per_slot_matches, joint_top_k, mixture_np)
            result["stage"] = "bypass_both_fixed"
            result["strategy"] = "bypass_both"
            return result

        # STRATEGY 2: GLOBAL_SCRUTINY
        all_pairs = self._get_top_n_pairs(per_slot_matches, self.top_n_pairs)
        expanded_pairs = self._residual_refine_all_pairs(
            mixture_np,
            all_pairs,
            fixed_components
        )
        joint_top_k = self._select_top_k_pairs(
            expanded_pairs,
            mixture_np,
            top_k
        )

        result = self._build_result(per_slot_matches, joint_top_k, mixture_np)
        result["stage"] = "global_scrutiny_both"
        result["strategy"] = "global_scrutiny"
        return result

    def _get_top_n_pairs(
        self,
        per_slot_matches: List[List[Tuple[str, float, int]]],
        n: int
    ) -> List[Tuple[int, int, float, float]]:
        """Get top N pairs from per-slot matches."""
        pairs = []
        for match_a in per_slot_matches[0][:n]:
            for match_b in per_slot_matches[1][:n]:
                smiles_a, score_a, idx_a = match_a
                smiles_b, score_b, idx_b = match_b
                if idx_a != idx_b:  # Avoid duplicate molecules
                    pairs.append((idx_a, idx_b, score_a, score_b))
        return pairs

    def _residual_refine_all_pairs(
        self,
        mixture_np: np.ndarray,
        pairs: List[Tuple[int, int, float, float]],
        fixed_components: List[bool],
    ) -> List[Tuple[int, int, float, float]]:
        """Apply residual refinement to all pairs."""
        expanded = list(pairs)  # Keep original pairs

        for idx_a, idx_b, score_a, score_b in pairs:
            # Refine slot 0 (A) if not fixed
            if not fixed_components[0]:
                residual = np.clip(
                    mixture_np - 0.5 * self._db_spectra_np[idx_b],
                    0,
                    None
                )
                residual_t = torch.from_numpy(residual.astype(np.float32))
                new_matches_a = self.database.find_best_match(
                    residual_t,
                    top_k=self.top_n_pairs
                )
                # Create new pairs with refined A and original B
                for smiles, new_score_a, new_idx_a in new_matches_a:
                    if new_idx_a != idx_b:
                        expanded.append((new_idx_a, idx_b, new_score_a, score_b))

            # Refine slot 1 (B) if not fixed
            if not fixed_components[1]:
                residual = np.clip(
                    mixture_np - 0.5 * self._db_spectra_np[idx_a],
                    0,
                    None
                )
                residual_t = torch.from_numpy(residual.astype(np.float32))
                new_matches_b = self.database.find_best_match(
                    residual_t,
                    top_k=self.top_n_pairs
                )
                # Create new pairs with original A and refined B
                for smiles, new_score_b, new_idx_b in new_matches_b:
                    if idx_a != new_idx_b:
                        expanded.append((idx_a, new_idx_b, score_a, new_score_b))

        return expanded

    def _select_top_k_pairs(
        self,
        pairs: List[Tuple[int, int, float, float]],
        mixture_np: np.ndarray,
        top_k: int,
    ) -> List[Tuple[Tuple[str, str], float]]:
        """Score all pairs and select top-k."""
        scored_pairs = []
        smiles_list = self.database.smiles
        seen_pairs = set()

        for idx_a, idx_b, score_a, score_b in pairs:
            pair_key = tuple(sorted([idx_a, idx_b]))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            # Compute reconstruction cosine
            spec_a = self._db_spectra_np[idx_a]
            spec_b = self._db_spectra_np[idx_b]
            reconstruction = 0.5 * (spec_a + spec_b)

            rec_norm = np.linalg.norm(reconstruction)
            mix_norm = np.linalg.norm(mixture_np)

            if rec_norm > 1e-12 and mix_norm > 1e-12:
                cos_recon = np.dot(reconstruction, mixture_np) / (rec_norm * mix_norm)
            else:
                cos_recon = 0.0

            # Component cosine average
            cos_comp = (score_a + score_b) / 2.0

            # Joint score
            joint_score = self.recon_weight * cos_recon + (1 - self.recon_weight) * cos_comp

            pair_smiles = (smiles_list[idx_a], smiles_list[idx_b])
            scored_pairs.append((pair_smiles, float(joint_score)))

        # Sort by score and return top-k
        scored_pairs.sort(key=lambda x: x[1], reverse=True)
        return scored_pairs[:top_k]

    def _build_cosine_pairs(
        self,
        per_slot_matches: List[List[Tuple[str, float, int]]],
        top_k: int,
    ) -> List[Tuple[Tuple[str, str], float]]:
        """Build pairs from cosine retrieval results."""
        pairs = []
        for match_a in per_slot_matches[0][:top_k]:
            for match_b in per_slot_matches[1][:top_k]:
                smiles_a, score_a, idx_a = match_a
                smiles_b, score_b, idx_b = match_b
                if idx_a != idx_b:
                    pair = (smiles_a, smiles_b)
                    avg_score = (score_a + score_b) / 2.0
                    pairs.append((pair, avg_score))

        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs[:top_k]

    def _build_result(
        self,
        per_slot_matches: List[List[Tuple[str, float, int]]],
        joint_top_k: List[Tuple[Tuple[str, str], float]],
        mixture_np: np.ndarray,
    ) -> Dict:
        """Build result dictionary."""
        if not joint_top_k:
            # Fallback to cosine retrieval
            joint_top_k = self._build_cosine_pairs(per_slot_matches, 1)

        top_pair = joint_top_k[0][0]
        top_score = joint_top_k[0][1]

        # Compute reconstruction cosine for metrics
        smiles_list = self.database.smiles
        indices = []
        for smi in top_pair:
            if smi in smiles_list:
                indices.append(smiles_list.index(smi))

        recon_cos = 0.0
        if len(indices) == 2:
            rec = 0.5 * (self._db_spectra_np[indices[0]] + self._db_spectra_np[indices[1]])
            n1, n2 = np.linalg.norm(rec), np.linalg.norm(mixture_np)
            if n1 > 1e-12 and n2 > 1e-12:
                recon_cos = float(np.dot(rec, mixture_np) / (n1 * n2))

        return {
            "per_slot_matches": per_slot_matches,
            "joint_top_k": joint_top_k,
            "joint_pair": top_pair,
            "joint_score": top_score,
            "reconstruction_cosine": recon_cos,
        }


def evaluate_smiles_matching(
    model,
    loader,
    database: LiquidIRDatabase,
    device: torch.device,
    cfg,
    max_samples: Optional[int] = None
) -> Dict:
    """Evaluate model on SMILES matching task."""
    model.eval()

    all_results = []
    all_correct = []
    all_similarities = []

    sample_count = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_samples and sample_count >= max_samples:
                break

            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            # Get model predictions
            outputs = model(mixture=batch["mixture"], stage=2)
            predicted_components = outputs["components"]  # [B, K, L]
            ground_truth_components = batch["components"]

            # Get ground truth SMILES from dataset
            batch_size = len(batch["mixture"])
            batch_start_idx = batch_idx * loader.batch_size
            batch_indices = list(range(batch_start_idx, batch_start_idx + batch_size))

            gt_smiles = None
            if hasattr(loader.dataset, 'get_smiles'):
                gt_smiles = [loader.dataset.get_smiles(idx) for idx in batch_indices]

            aligned_components = []
            for sample_idx in range(batch_size):
                aligned_predicted, _, _ = align_components_to_ground_truth(
                    predicted_components[sample_idx],
                    ground_truth_components[sample_idx],
                )
                aligned_components.append(aligned_predicted)
            aligned_components = torch.stack(aligned_components)

            # Match to SMILES
            matching_results = match_components_to_smiles(
                aligned_components,
                database,
                ground_truth_smiles=gt_smiles,
            )

            # Store results
            all_results.extend(matching_results['matched_smiles'])
            all_similarities.extend(matching_results['similarities'])

            if 'correct_matches' in matching_results:
                all_correct.extend(matching_results['correct_matches'])

            sample_count += len(batch["mixture"])

    # Compute aggregate metrics
    metrics = {
        'n_samples': sample_count,
        'mean_similarity': np.mean([np.mean(sims) for sims in all_similarities]),
    }

    if all_correct:
        all_correct_array = np.array(all_correct)  # [B, K]
        metrics['top1_accuracy'] = all_correct_array.mean()
        metrics['component_accuracy'] = all_correct_array.mean(axis=0).tolist()  # Per-component accuracy
        metrics['exact_match_rate'] = (all_correct_array.all(axis=1).mean())  # Both components correct

    return {
        'metrics': metrics,
        'results': all_results,
        'similarities': all_similarities,
        'correct': all_correct if all_correct else None
    }

