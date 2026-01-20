"""Main training pipeline for the project.

This script implements the core multi-stage training pipeline in a concise and
maintainable manner. The pipeline:

1. Configures logging and loads all AnnData datasets declared in ``config.py``.
2. Applies lightweight preprocessing (filtering, normalisation, HVG selection).
3. Performs feature engineering (PCA + simple pathway scores) and clustering.
4. Builds joint cell/drug representations using dataset-specific fingerprints.
5. Trains per-cluster ProtoMechanism models with class-balancing weights.
6. Tunes decision thresholds, evaluates performance, and records results.
7. Extracts prototype diagnostics for downstream reporting.
"""


from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, StratifiedKFold, StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_class_weight

try:
    from scipy import sparse
except ImportError:  # pragma: no cover
    sparse = None

import scanpy as sc  # type: ignore

sys.path.append(".")
sys.path.append("code")
from config import CONFIG  # noqa: E402
from data_loader import (  # noqa: E402
    get_results_manager,
    load_single_cell_dataset,
)
from features.pipeline import build_feature_matrix  # noqa: E402
from features.drug_fingerprints import (  # noqa: E402
    DrugFingerprintCache,
    load_cache_for_flags,
)
from postprocess import learn_thresholds  # noqa: E402
from models import build_model  # noqa: E402

ADAPTIVE_FLAG_REGISTRY: Dict[str, Dict[str, object]] = {}
PATHWAY_PREFIXES: Tuple[str, ...] = ("ssgsea_", "progeny_", "reactome_")


def _has_prefix(name: object, prefixes: Tuple[str, ...]) -> bool:
    token = str(name).lower()
    return any(token.startswith(prefix) for prefix in prefixes)


def _is_pathway_feature(name: object) -> bool:
    return _has_prefix(name, PATHWAY_PREFIXES)


@dataclass
class DatasetBundle:
    """Container for all artefacts derived from a single dataset."""

    name: str
    adata: "sc.AnnData"
    features: np.ndarray
    feature_names: List[str]
    labels: np.ndarray
    clusters: np.ndarray
    metadata: Dict[str, object]
    sample_weights: Optional[np.ndarray] = None
    groups: Optional[np.ndarray] = None
    prototypes: Optional[Dict[str, Dict[str, np.ndarray]]] = None
    drug_fingerprint: Optional[np.ndarray] = None
    drug_fingerprint_names: Optional[List[str]] = None
    pathway_feature_indices: Optional[List[int]] = None
    drug_feature_indices: Optional[List[int]] = None


FINGERPRINT_CACHE: Optional[DrugFingerprintCache] = None
FINGERPRINT_CACHE_PATH: Optional[str] = None


# --------------------------------------------------------------------------- #
#                               Helper utilities                              #
# --------------------------------------------------------------------------- #


def _to_dense(matrix) -> np.ndarray:
    if sparse is not None and sparse.issparse(matrix):
        return matrix.toarray()
    return np.asarray(matrix)


def _get_fingerprint_cache(flags: Mapping[str, object]) -> Optional[DrugFingerprintCache]:
    global FINGERPRINT_CACHE, FINGERPRINT_CACHE_PATH
    cache_path = flags.get("drug_fingerprint_cache")
    if not cache_path:
        return None
    cache_str = str(cache_path)
    if FINGERPRINT_CACHE is None or FINGERPRINT_CACHE_PATH != cache_str:
        FINGERPRINT_CACHE = load_cache_for_flags(flags)
        FINGERPRINT_CACHE_PATH = cache_str if FINGERPRINT_CACHE is not None else None
    return FINGERPRINT_CACHE


def infer_group_labels(adata: "sc.AnnData", dataset_name: str) -> np.ndarray:
    """Infer grouping labels to keep related cells together during splits."""
    candidates = [
        "patient_id",
        "patient",
        "donor",
        "sample_id",
        "sample",
        "orig.ident",
        "batch",
        "library_id",
    ]
    for column in candidates:
        if column in adata.obs and adata.obs[column].nunique(dropna=True) > 1:
            values = adata.obs[column].astype(str).fillna("NA").to_numpy()
            logger.debug("{} grouping column '{}' with {} groups.", dataset_name, column, len(np.unique(values)))
            return values

    obs_names = adata.obs_names.astype(str)
    prefixes = np.array([name.split("_")[0] for name in obs_names])
    if len(np.unique(prefixes)) > 1:
        logger.debug("{} grouping inferred from obs name prefixes.", dataset_name)
        return prefixes

    max_chunk = 500
    n_groups = max(2, int(np.ceil(adata.n_obs / max_chunk)))
    indices = np.arange(adata.n_obs)
    shuffled = np.random.permutation(indices)
    groups = np.empty(adata.n_obs, dtype=object)
    for gid, chunk in enumerate(np.array_split(shuffled, n_groups)):
        groups[chunk] = f"chunk_{gid}"
    logger.warning(
        "{} fell back to synthetic grouping ({} groups); consider providing patient/sample identifiers.",
        dataset_name,
        len(np.unique(groups)),
    )
    return groups


def setup_logging(
    datasets_config: Mapping[str, Mapping[str, object]],
    *,
    run_name: str = "run",
) -> Path:
    """Configure loguru sinks for the run and per-dataset log files."""
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    base_dir = (Path(CONFIG["log_root"]).expanduser() / run_name).resolve()
    base_dir.mkdir(parents=True, exist_ok=True)

    for dataset_name in datasets_config:
        log_file = base_dir / f"{dataset_name}.log"
        logger.add(log_file, level="INFO", rotation="5 MB", enqueue=True)

    return base_dir


def load_multi_datasets(
    datasets_config: Mapping[str, Mapping[str, object]],
) -> Dict[str, "sc.AnnData"]:
    """Load all configured AnnData targets and ensure metadata is attached."""
    datasets: Dict[str, "sc.AnnData"] = {}
    for dataset_name, dataset_config in datasets_config.items():
        adata = load_single_cell_dataset(dataset_name, CONFIG, copy=True)
        adata.obs["dataset_name"] = dataset_name
        adata.uns["dataset_config"] = dict(dataset_config)
        datasets[dataset_name] = adata
        logger.info("Loaded {} with shape {}", dataset_name, adata.shape)
    return datasets


def resolve_pipeline_flags(dataset_name: Optional[str] = None) -> Dict[str, object]:
    """Return pipeline flags merged with per-dataset overrides."""
    base = dict(CONFIG.get("pipeline_flags", {}))
    overrides = CONFIG.get("dataset_flag_overrides", {}) or {}
    if dataset_name and dataset_name in overrides:
        base.update(overrides[dataset_name])
    if dataset_name and dataset_name in ADAPTIVE_FLAG_REGISTRY:
        base.update(ADAPTIVE_FLAG_REGISTRY[dataset_name])
    return base


def _compute_adaptive_overrides(bundle: DatasetBundle) -> Dict[str, object]:
    """Heuristically adjust RL/epoch hyperparameters based on dataset profile."""
    n_samples = bundle.features.shape[0]
    n_clusters = len(np.unique(bundle.clusters))
    pos_rate = float(np.mean(bundle.labels)) if bundle.labels.size else 0.5
    base = CONFIG.get("pipeline_flags", {})

    overrides: Dict[str, object] = {}
    small_threshold = 15000
    large_threshold = 40000

    if n_samples <= small_threshold:
        overrides["prototype_epochs"] = min(int(base.get("prototype_epochs", 75)), 60)
        overrides["gating_rl_warmup_epochs"] = max(int(base.get("gating_rl_warmup_epochs", 0)), 5)
    elif n_samples >= large_threshold or n_clusters >= 30:
        overrides["prototype_epochs"] = max(int(base.get("prototype_epochs", 75)), 90)
        overrides["gating_rl_eta"] = min(float(base.get("gating_rl_eta", 0.1)), 0.08)
        overrides["gating_rl_warmup_epochs"] = max(int(base.get("gating_rl_warmup_epochs", 0)), 3)

    imbalance = min(pos_rate, 1.0 - pos_rate)
    if imbalance < 0.2:
        overrides["gating_rl_eta"] = min(float(overrides.get("gating_rl_eta", base.get("gating_rl_eta", 0.1))), 0.08)
        overrides["gating_rl_entropy_coef"] = max(float(base.get("gating_rl_entropy_coef", 0.001)), 0.002)

    return overrides


def _token_variants(token: str) -> List[str]:
    cleaned = token.strip().lower()
    cleaned = cleaned.replace("β", "beta").replace("γ", "gamma").replace("δ", "delta")
    replacements = set()
    candidates = {
        cleaned,
        cleaned.replace("-", "_"),
        cleaned.replace("_", "-"),
        cleaned.replace(" ", "_"),
        cleaned.replace(" ", "-"),
        cleaned.replace("/", "_"),
        cleaned.replace("/", "-"),
    }
    for cand in candidates:
        if cand:
            replacements.add(cand)
    return list(replacements)


def _feature_priority(name: str) -> int:
    lowered = name.lower()
    if lowered.startswith("progeny_"):
        return 0
    if lowered.startswith("ssgsea_hallmark") or "hallmark" in lowered:
        return 1
    if lowered.startswith("ssgsea_"):
        return 2
    if lowered.startswith("gsva_"):
        return 3
    return 4


def _compute_proto_prior_overrides(bundle: DatasetBundle) -> Dict[str, object]:
    """Map configured pathway keywords to feature indices for proto_mech."""
    base_flags = CONFIG.get("pipeline_flags", {})
    priors = base_flags.get("proto_pathway_priors") or {}
    if not priors:
        return {}

    feature_names = list(bundle.feature_names)
    if not feature_names:
        return {}

    lowered = [str(name).lower() for name in feature_names]
    n_features = len(feature_names)
    index_map: Dict[int, List[int]] = {}
    used_indices: set[int] = set()

    def _proto_sort_key(value):
        text = str(value)
        if text.isdigit():
            return (0, int(text))
        return (1, text)

    for proto_key in sorted(priors, key=_proto_sort_key):
        tokens = priors.get(proto_key)
        try:
            proto_idx = int(proto_key)
        except (TypeError, ValueError):
            continue
        if proto_idx < 0:
            continue
        if not tokens:
            continue
        if not isinstance(tokens, (list, tuple, set)):
            token_list = [tokens]
        else:
            token_list = list(tokens)

        matched: List[int] = []
        for token in token_list:
            if token is None:
                continue
            if isinstance(token, int):
                if 0 <= token < n_features and token not in used_indices:
                    matched.append(int(token))
                    used_indices.add(int(token))
                continue
            token_str = str(token).strip()
            if not token_str:
                continue
            if token_str.isdigit():
                idx_val = int(token_str)
                if 0 <= idx_val < n_features and idx_val not in used_indices:
                    matched.append(idx_val)
                    used_indices.add(idx_val)
                    continue
            variants = _token_variants(token_str)
            local_matches: List[int] = []
            for idx, name in enumerate(lowered):
                if idx in used_indices:
                    continue
                if any(variant in name for variant in variants):
                    local_matches.append(idx)
            if local_matches:
                local_matches.sort(key=lambda i: (_feature_priority(feature_names[i]), i))
                for idx in local_matches:
                    if idx not in used_indices:
                        matched.append(idx)
                        used_indices.add(idx)
        if matched:
            if len(matched) < 5:
                logger.warning(
                    "Prototype prior {} matched only {} features ({}) in dataset {}; consider expanding keywords.",
                    proto_idx,
                    len(matched),
                    ", ".join(feature_names[idx] for idx in matched[:5]),
                    bundle.name,
                )
            index_map[proto_idx] = matched

    if not index_map:
        return {}

    return {
        "proto_prior_indices": {str(key): value for key, value in index_map.items()},
        "proto_feature_names": feature_names,
    }


def register_adaptive_flags(bundle: DatasetBundle) -> None:
    overrides = _compute_adaptive_overrides(bundle)
    proto_overrides = _compute_proto_prior_overrides(bundle)
    if proto_overrides:
        overrides.update(proto_overrides)
    if overrides:
        ADAPTIVE_FLAG_REGISTRY[bundle.name] = overrides
        logger.info(
            "Adaptive flags for {} (n_samples={}, n_clusters={}): {}",
            bundle.name,
            bundle.features.shape[0],
            len(np.unique(bundle.clusters)),
            overrides,
        )


def get_sensitive_mask(adata: "sc.AnnData") -> np.ndarray:
    """Return a boolean mask indicating sensitive (positive) samples."""
    if "standardized_response" not in adata.obs:
        raise KeyError("Missing 'standardized_response' column.")
    response = adata.obs["standardized_response"].astype(int).to_numpy()
    return response == 1


def extract_drug_functional_fingerprints(
    datasets: Mapping[str, "sc.AnnData"],
) -> Dict[str, Dict[str, np.ndarray]]:
    """Obtain simple differential-expression fingerprints per dataset."""
    fingerprints: Dict[str, Dict[str, np.ndarray]] = {}
    for dataset_name, adata in datasets.items():
        try:
            sensitive_mask = get_sensitive_mask(adata)
        except KeyError as exc:  # pragma: no cover - defensive
            logger.warning("{} skipped for fingerprints: {}", dataset_name, exc)
            continue

        resistant_mask = ~sensitive_mask
        if sensitive_mask.sum() == 0 or resistant_mask.sum() == 0:
            logger.warning(
                "{} lacks both response classes; fingerprint skipped.", dataset_name
            )
            continue

        matrix = _to_dense(adata.X)
        sensitive_mean = matrix[sensitive_mask].mean(axis=0)
        resistant_mean = matrix[resistant_mask].mean(axis=0)
        diff_vector = sensitive_mean - resistant_mean
        norm = float(np.linalg.norm(diff_vector)) or 1.0

        fingerprints[dataset_name] = {
            "vector": diff_vector / norm,
            "genes": adata.var_names.tolist(),
        }

        logger.debug(
            "{} fingerprint computed over {} genes (norm {:.4f}).",
            dataset_name,
            len(fingerprints[dataset_name]["genes"]),
            norm,
        )

    return fingerprints


def standardize_multi_dataset_responses(
    datasets: Mapping[str, "sc.AnnData"],
) -> Tuple[Dict[str, "sc.AnnData"], Dict[str, Dict[str, int]]]:
    """Ensure binary labels exist and collect simple distribution statistics."""
    standardized: Dict[str, "sc.AnnData"] = {}
    summary: Dict[str, Dict[str, int]] = {}
    for dataset_name, adata in datasets.items():
        mask = get_sensitive_mask(adata)
        adata.obs["standardized_response"] = mask.astype(int)
        standardized[dataset_name] = adata
        summary[dataset_name] = {
            "n_sensitive": int(mask.sum()),
            "n_resistant": int(len(mask) - mask.sum()),
        }
        logger.info(
            "{} label balance -> sensitive: {}, resistant: {}",
            dataset_name,
            summary[dataset_name]["n_sensitive"],
            summary[dataset_name]["n_resistant"],
        )
    return standardized, summary


def basic_preprocessing(
    adata: "sc.AnnData",
    dataset_name: str,
    min_genes: int = 200,
    min_cells: int = 3,
    min_counts: int = 200,
    max_mito_pct: float = 20.0,
) -> "sc.AnnData":
    """Filter low-quality cells and genes (counts / genes / mito%)."""
    filtered = adata.copy()
    init_cells, init_genes = filtered.n_obs, filtered.n_vars

    # 计算线粒体占比
    try:
        filtered.var["mt"] = filtered.var_names.str.upper().str.startswith("MT-")
        sc.pp.calculate_qc_metrics(filtered, qc_vars=["mt"], inplace=True)
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.warning("%s QC metric calculation skipped (mt): %s", dataset_name, exc)

    if filtered.n_obs >= min_genes:
        sc.pp.filter_cells(filtered, min_genes=min_genes)
    if min_counts and filtered.n_obs:
        sc.pp.filter_cells(filtered, min_counts=min_counts)
    if "pct_counts_mt" in filtered.obs and max_mito_pct is not None:
        filtered = filtered[filtered.obs["pct_counts_mt"] <= max_mito_pct].copy()
    if filtered.n_vars >= min_cells:
        sc.pp.filter_genes(filtered, min_cells=min_cells)

    logger.debug(
        "%s QC -> cells %d -> %d, genes %d -> %d (min_genes=%d, min_counts=%d, mito<=%.1f%%)",
        dataset_name,
        init_cells,
        filtered.n_obs,
        init_genes,
        filtered.n_vars,
        min_genes,
        min_counts,
        max_mito_pct if max_mito_pct is not None else float("nan"),
    )
    return filtered


def clean_expression_data(
    adata: "sc.AnnData",
    dataset_name: str,
    *,
    min_genes: int = 200,
    min_cells: int = 3,
    min_counts: int = 200,
    max_mito_pct: float = 20.0,
) -> Tuple["sc.AnnData", Dict[str, int]]:
    """Apply basic filtering and report the remaining dimensions."""
    cleaned = basic_preprocessing(
        adata,
        dataset_name,
        min_genes=min_genes,
        min_cells=min_cells,
        min_counts=min_counts,
        max_mito_pct=max_mito_pct,
    )
    stats = {"n_cells": int(cleaned.n_obs), "n_genes": int(cleaned.n_vars)}
    return cleaned, stats


def normalize_and_log_transform(
    adata: "sc.AnnData",
    dataset_name: str,
    target_sum: float = 10_000.0,
) -> "sc.AnnData":
    """Library-size normalisation followed by log1p transform."""
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    logger.debug("{} normalised with target sum {}", dataset_name, target_sum)
    return adata


def select_highly_variable_genes(
    adata: "sc.AnnData",
    dataset_name: str,
    n_top_genes: int = 2000,
) -> Tuple["sc.AnnData", Dict[str, int]]:
    """Retain highly variable genes and compute PCA for downstream steps."""
    n_top = min(n_top_genes, adata.n_vars)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top, subset=True, flavor="seurat_v3")
    sc.pp.scale(adata, max_value=10)

    n_components = min(50, max(2, adata.n_vars - 1))
    sc.tl.pca(adata, n_comps=n_components, svd_solver="arpack")

    logger.debug(
        "{} HVG selection retained {} genes ({} PCA components).",
        dataset_name,
        adata.n_vars,
        n_components,
    )
    return adata, {"n_hvg": int(adata.n_vars), "pca_components": int(n_components)}


def preprocess_multi_datasets(
    datasets: Mapping[str, "sc.AnnData"],
) -> Tuple[Dict[str, "sc.AnnData"], Dict[str, Dict[str, int]]]:
    """Execute the preprocessing stack for every dataset."""
    processed: Dict[str, "sc.AnnData"] = {}
    stats: Dict[str, Dict[str, int]] = {}
    flags = CONFIG.get("pipeline_flags", {})
    qc_min_genes = int(flags.get("qc_min_genes", 200))
    qc_min_counts = int(flags.get("qc_min_counts", 200))
    qc_max_mito_pct = float(flags.get("qc_max_mito_pct", 20.0))
    qc_min_cells = int(flags.get("qc_min_cells", 3))

    for dataset_name, adata in datasets.items():
        cleaned, clean_stats = clean_expression_data(
            adata,
            dataset_name,
            min_genes=qc_min_genes,
            min_cells=qc_min_cells,
            min_counts=qc_min_counts,
            max_mito_pct=qc_max_mito_pct,
        )
        normalized = normalize_and_log_transform(cleaned, dataset_name)
        processed_adata, hv_stats = select_highly_variable_genes(normalized, dataset_name)
        processed[dataset_name] = processed_adata
        stats[dataset_name] = {**clean_stats, **hv_stats}

    return processed, stats


def enhanced_feature_engineering_single(
    adata: "sc.AnnData",
    dataset_name: str,
) -> Tuple["sc.AnnData", Dict[str, float]]:
    """Augment AnnData with engineered representations."""
    engineered = adata.copy()
    feature_matrix, feature_names, diagnostics = build_feature_matrix(engineered, CONFIG)

    engineered.obsm["engineered_features"] = feature_matrix
    engineered.uns["engineered_feature_names"] = feature_names

    stats = {
        **diagnostics,
        "engineered_dim": int(feature_matrix.shape[1]),
    }

    logger.debug(
        "%s feature matrix with %d features (mode=%s)",
        dataset_name,
        feature_matrix.shape[1],
        diagnostics.get("mode", "unknown"),
    )
    return engineered, stats


def enhanced_feature_engineering_multi(
    datasets: Mapping[str, "sc.AnnData"],
) -> Tuple[Dict[str, "sc.AnnData"], Dict[str, Dict[str, float]]]:
    """Apply feature engineering to all datasets."""
    engineered: Dict[str, "sc.AnnData"] = {}
    stats: Dict[str, Dict[str, float]] = {}
    for dataset_name, adata in datasets.items():
        engineered_dataset, dataset_stats = enhanced_feature_engineering_single(adata, dataset_name)
        engineered[dataset_name] = engineered_dataset
        stats[dataset_name] = dataset_stats
    return engineered, stats


def perform_clustering_single(
    adata: "sc.AnnData",
    dataset_name: str,
    n_neighbors: int = 15,
    resolution: float = 0.5,
) -> Tuple["sc.AnnData", Dict[str, object]]:
    """Cluster cells using Leiden on the PCA representation."""
    clustered = adata.copy()
    if clustered.n_obs < 3:
        clustered.obs["cluster"] = np.full(clustered.n_obs, "0")
        return clustered, {"n_clusters": 1, "cluster_sizes": {"0": int(clustered.n_obs)}}

    neighbor_count = min(n_neighbors, max(2, clustered.n_obs - 1))
    sc.pp.neighbors(clustered, n_neighbors=neighbor_count, use_rep="X_pca")
    sc.tl.leiden(clustered, resolution=resolution, key_added="cluster")
    clustered.obs["cluster"] = clustered.obs["cluster"].astype(str)

    counts = clustered.obs["cluster"].value_counts().to_dict()
    stats = {"n_clusters": len(counts), "cluster_sizes": {k: int(v) for k, v in counts.items()}}

    logger.debug(
        "{} clustering produced {} clusters (min={}, max={}).",
        dataset_name,
        stats["n_clusters"],
        min(counts.values()),
        max(counts.values()),
    )
    return clustered, stats


def cell_clustering_and_subgroups(
    datasets: Mapping[str, "sc.AnnData"],
) -> Tuple[Dict[str, "sc.AnnData"], Dict[str, Dict[str, object]]]:
    """Run clustering for every dataset."""
    clustered: Dict[str, "sc.AnnData"] = {}
    stats: Dict[str, Dict[str, object]] = {}
    for dataset_name, adata in datasets.items():
        clustered_dataset, dataset_stats = perform_clustering_single(adata, dataset_name)
        clustered[dataset_name] = clustered_dataset
        stats[dataset_name] = dataset_stats
    return clustered, stats


def _build_fingerprint_weights(
    adata: "sc.AnnData",
    fingerprint_info: Dict[str, np.ndarray],
) -> np.ndarray:
    gene_to_weight = dict(zip(fingerprint_info["genes"], fingerprint_info["vector"]))
    return np.array([gene_to_weight.get(gene, 0.0) for gene in adata.var_names], dtype=float)


def _compute_fingerprint_score(matrix, weights: np.ndarray) -> np.ndarray:
    if sparse is not None and sparse.issparse(matrix):
        score = matrix.dot(weights)
    else:
        score = matrix @ weights
    return np.asarray(score).ravel()


def _extract_feature_block_indices(feature_names: Sequence[str]) -> Dict[str, np.ndarray]:
    """Identify indices for gene-only and pathway-specific features."""
    pathway_tokens = ("ssgsea_", "gsva_", "progeny_")
    pathway_idx_list = [
        idx for idx, name in enumerate(feature_names) if str(name).startswith(pathway_tokens)
    ]
    pathway_idx = np.array(pathway_idx_list, dtype=int)
    pathway_set = set(pathway_idx_list)
    gene_idx = np.array(
        [idx for idx in range(len(feature_names)) if idx not in pathway_set],
        dtype=int,
    )
    return {
        "all": np.arange(len(feature_names), dtype=int),
        "gene": gene_idx,
        "pathway": pathway_idx,
    }


def _compute_bundle_prototypes(
    features: np.ndarray,
    feature_names: Sequence[str],
    clusters: Sequence[str],
) -> Dict[str, Dict[str, np.ndarray]]:
    """Compute per-cluster prototypes for downstream model initialisation."""
    indices = _extract_feature_block_indices(feature_names)
    cluster_ids = np.asarray(clusters, dtype=str)
    prototypes: Dict[str, Dict[str, np.ndarray]] = {
        "clusters": {},
        "global": {},
        "feature_blocks": {
            key: value.tolist() for key, value in indices.items()
        },
    }

    # Global prototypes
    prototypes["global"]["all"] = features.mean(axis=0)
    if indices["gene"].size > 0:
        prototypes["global"]["gene"] = features[:, indices["gene"]].mean(axis=0)
    if indices["pathway"].size > 0:
        prototypes["global"]["pathway"] = features[:, indices["pathway"]].mean(axis=0)

    for cluster_id in np.unique(cluster_ids):
        mask = cluster_ids == cluster_id
        if not mask.any():
            continue
        cluster_features = features[mask]
        cluster_entry: Dict[str, np.ndarray] = {
            "all": cluster_features.mean(axis=0),
        }
        if indices["gene"].size > 0:
            cluster_entry["gene"] = cluster_features[:, indices["gene"]].mean(axis=0)
        if indices["pathway"].size > 0:
            cluster_entry["pathway"] = cluster_features[:, indices["pathway"]].mean(axis=0)
        prototypes["clusters"][str(cluster_id)] = cluster_entry

    return prototypes


def construct_joint_features_single(
    adata: "sc.AnnData",
    dataset_name: str,
    fingerprint_info: Dict[str, np.ndarray] | None,
    fusion_strategy: str = "concat",
    *,
    pipeline_flags: Optional[Mapping[str, object]] = None,
) -> Tuple[DatasetBundle, Dict[str, int]]:
    """Create the dataset bundle containing features, labels, and metadata."""
    flags = pipeline_flags or CONFIG.get("pipeline_flags", {})
    if "engineered_features" in adata.obsm:
        base_features = adata.obsm["engineered_features"]
        base_names = list(adata.uns.get("engineered_feature_names", []))
    else:
        base_features = adata.obsm.get("X_pca", _to_dense(adata.X))
        base_names = [f"pca_{idx + 1}" for idx in range(base_features.shape[1])]

    extra_components: List[np.ndarray] = []
    extra_names: List[str] = []

    fingerprint_vector: Optional[np.ndarray] = None
    fingerprint_names: Optional[List[str]] = None
    if flags.get("enable_drug_fingerprints"):
        cache = _get_fingerprint_cache(flags)
        drug_name = adata.uns.get("dataset_config", {}).get("drug_name")
        if cache and drug_name:
            fp_static_weight = float(flags.get("drug_fingerprint_static_weight", 0.5))
            fp_dynamic_weight = float(flags.get("drug_fingerprint_dynamic_weight", 0.5))
            fp_vector = cache.get(
                drug_name,
                static_weight=fp_static_weight,
                dynamic_weight=fp_dynamic_weight,
            )
            if fp_vector:
                fingerprint_vector = fp_vector.values
                fingerprint_names = fp_vector.feature_names
                if fingerprint_vector.size > 0:
                    tiled = np.repeat(fingerprint_vector.reshape(1, -1), base_features.shape[0], axis=0)
                    base_features = np.hstack([base_features, tiled])
                    base_names.extend(fingerprint_names)
                dataset_meta = dict(adata.uns.get("dataset_config", {}))
                if fp_vector.pathway_profile:
                    dataset_meta["drug_pathway_profile"] = fp_vector.pathway_profile
                if fp_vector.static_tokens:
                    dataset_meta["drug_static_tokens"] = fp_vector.static_tokens
                adata.uns["dataset_config"] = dataset_meta
        elif flags.get("enable_drug_fingerprints"):
            logger.warning(
                "Drug fingerprint requested for %s but cache unavailable or drug name missing.",
                dataset_name,
            )

    if fusion_strategy != "concat":  # pragma: no cover - defensive
        raise ValueError(f"Unsupported fusion strategy: {fusion_strategy}")

    features = (
        np.hstack([base_features, *extra_components]) if extra_components else base_features
    )
    feature_names = base_names + extra_names
    pathway_indices = [idx for idx, name in enumerate(feature_names) if _is_pathway_feature(name)]
    drug_indices = None
    if fingerprint_names:
        fingerprint_set = {name for name in fingerprint_names}
        drug_indices = [idx for idx, name in enumerate(feature_names) if name in fingerprint_set]

    labels = adata.obs["standardized_response"].astype(int).to_numpy()
    clusters = (
        adata.obs["cluster"].astype(str).to_numpy()
        if "cluster" in adata.obs
        else np.full(adata.n_obs, "0", dtype=str)
    )
    metadata = dict(adata.uns.get("dataset_config", {}))
    metadata.setdefault("drug_name", metadata.get("drug_name", "unknown"))
    metadata.setdefault("label_strategy", metadata.get("label_strategy", "unknown"))
    metadata.setdefault("geneset", metadata.get("geneset", "all"))
    metadata["n_target"] = int(adata.n_obs)
    metadata["genes"] = int(adata.n_vars)
    metadata["feature_dim"] = int(features.shape[1])
    metadata["pathway_feature_indices"] = pathway_indices
    if drug_indices is not None:
        metadata["drug_fingerprint_indices"] = drug_indices
    if fingerprint_vector is not None and fingerprint_names:
        metadata["drug_fingerprint_names"] = fingerprint_names
        metadata["drug_fingerprint_vector"] = fingerprint_vector.tolist()

    groups = infer_group_labels(adata, dataset_name)
    metadata["n_groups"] = int(len(np.unique(groups)))

    prototypes = _compute_bundle_prototypes(features, feature_names, clusters)

    bundle = DatasetBundle(
        name=dataset_name,
        adata=adata,
        features=features,
        feature_names=feature_names,
        labels=labels,
        clusters=clusters,
        metadata=metadata,
        groups=groups,
        prototypes=prototypes,
        drug_fingerprint=fingerprint_vector,
        drug_fingerprint_names=fingerprint_names,
        pathway_feature_indices=pathway_indices,
        drug_feature_indices=drug_indices,
    )

    stats = {
        "n_samples": int(features.shape[0]),
        "n_features": int(features.shape[1]),
        "n_cluster_prototypes": len(prototypes.get("clusters", {})),
    }
    return bundle, stats


def construct_cell_drug_joint_features(
    clustered_datasets: Mapping[str, "sc.AnnData"],
    drug_fingerprints: Mapping[str, Dict[str, np.ndarray]],
) -> Tuple[Dict[str, DatasetBundle], Dict[str, Dict[str, int]]]:
    """Build dataset bundles for downstream modelling."""
    joint_datasets: Dict[str, DatasetBundle] = {}
    stats: Dict[str, Dict[str, int]] = {}
    for dataset_name, adata in clustered_datasets.items():
        fingerprint = drug_fingerprints.get(dataset_name)
        flags = resolve_pipeline_flags(dataset_name)
        bundle, bundle_stats = construct_joint_features_single(
            adata,
            dataset_name,
            fingerprint,
            pipeline_flags=flags,
        )
        register_adaptive_flags(bundle)
        joint_datasets[dataset_name] = bundle
        stats[dataset_name] = bundle_stats
    return joint_datasets, stats


def handle_cluster_aware_imbalance(
    joint_datasets: Mapping[str, DatasetBundle],
) -> Tuple[Dict[str, DatasetBundle], Dict[str, Dict[str, float]]]:
    """Assign sample weights per cluster to mitigate class imbalance."""
    balanced: Dict[str, DatasetBundle] = {}
    stats: Dict[str, Dict[str, float]] = {}

    for dataset_name, bundle in joint_datasets.items():
        weights = np.ones_like(bundle.labels, dtype=float)
        balanced[dataset_name] = replace(bundle, sample_weights=weights)
        stats[dataset_name] = {"min_weight": 1.0, "max_weight": 1.0}

    return balanced, stats


def initialize_cluster_model(
    cluster_id: str,
    cluster_size: int,
    n_features: int,
    *,
    dataset_name: str | None = None,
    random_state: int = 42,
    feature_names: Optional[Sequence[str]] = None,
    pathway_indices: Optional[Sequence[int]] = None,
    drug_indices: Optional[Sequence[int]] = None,
) -> object:
    """Create a cluster-level classifier according to pipeline flags."""
    flags = resolve_pipeline_flags(dataset_name)
    model_type = flags.get("model_type", "proto_mech")
    model_config = dict(CONFIG)
    model_config["pipeline_flags"] = flags
    model = build_model(
        model_type,
        input_dim=n_features,
        random_state=random_state,
        config=model_config,
        feature_names=feature_names,
        pathway_indices=pathway_indices,
        drug_indices=drug_indices,
    )
    logger.debug(
        "%s cluster %s -> model '%s' (size=%d)",
        dataset_name or "dataset",
        cluster_id,
        model_type,
        cluster_size,
    )
    return model


def _t_critical(alpha: float, df: int) -> float:
    if df <= 0:
        return float("nan")
    try:
        from scipy.stats import t as student_t  # type: ignore
    except Exception:
        return 1.96
    return float(student_t.ppf(1.0 - alpha / 2.0, df))


def _t_confidence_interval(
    values: Sequence[float],
    *,
    alpha: float = 0.05,
) -> Tuple[float, float, float, float]:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    mean = float(np.nanmean(clean))
    if clean.size < 2:
        return mean, float("nan"), float("nan"), float("nan")
    std = float(np.nanstd(clean, ddof=1))
    tcrit = _t_critical(alpha, int(clean.size - 1))
    if not np.isfinite(std) or not np.isfinite(tcrit):
        return mean, std, float("nan"), float("nan")
    half = tcrit * std / math.sqrt(clean.size)
    return mean, std, mean - half, mean + half


def aggregate_metric_list(
    metrics: Iterable[Dict[str, float]],
    *,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """Summarise metric dictionaries into mean/std and 95% t-intervals."""
    metrics = list(metrics)
    if not metrics:
        return {}
    aggregate: Dict[str, float] = {}
    keys = metrics[0].keys()
    for key in keys:
        values = np.array([entry[key] for entry in metrics], dtype=float)
        mean, std, ci_low, ci_high = _t_confidence_interval(values, alpha=alpha)
        aggregate[f"mean_{key}"] = mean
        aggregate[f"std_{key}"] = std
        aggregate[f"ci_{key}_low"] = ci_low
        aggregate[f"ci_{key}_high"] = ci_high
    return aggregate


def compute_evaluation_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> Dict[str, float]:
    """Compute standard classification metrics."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    try:
        metrics["auc"] = float(roc_auc_score(y_true, y_proba))
    except ValueError:
        metrics["auc"] = float("nan")
    try:
        metrics["pr_auc"] = float(average_precision_score(y_true, y_proba))
    except ValueError:
        metrics["pr_auc"] = float("nan")
    return metrics


def bootstrap_evaluation_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    *,
    n_bootstrap: int = 500,
    seed: int = 42,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """Estimate test metric dispersion and 95% t-interval via bootstrap resampling."""
    keys = ("auc", "pr_auc", "accuracy", "precision", "recall", "f1")
    if y_true.size == 0:
        return {
            f"std_{key}": float("nan")
            for key in keys
        } | {
            f"ci_{key}_low": float("nan")
            for key in keys
        } | {
            f"ci_{key}_high": float("nan")
            for key in keys
        }

    rng = np.random.default_rng(seed)
    indices = np.arange(y_true.size)
    samples: Dict[str, list[float]] = {key: [] for key in keys}

    for _ in range(n_bootstrap):
        boot_idx = rng.choice(indices, size=y_true.size, replace=True)
        boot_metrics = compute_evaluation_metrics(y_true[boot_idx], y_pred[boot_idx], y_proba[boot_idx])
        for key in keys:
            value = boot_metrics.get(key, float("nan"))
            if np.isfinite(value):
                samples[key].append(float(value))

    ci: Dict[str, float] = {}
    for key in keys:
        values = np.asarray(samples[key], dtype=float)
        if values.size:
            _, std, ci_low, ci_high = _t_confidence_interval(values, alpha=alpha)
            ci[f"std_{key}"] = std
            ci[f"ci_{key}_low"] = ci_low
            ci[f"ci_{key}_high"] = ci_high
        else:
            ci[f"std_{key}"] = float("nan")
            ci[f"ci_{key}_low"] = float("nan")
            ci[f"ci_{key}_high"] = float("nan")
    return ci


def train_cluster_models_single(
    bundle: DatasetBundle,
    train_mask: np.ndarray,
    candidate_thresholds: Optional[Iterable[float]] = None,
) -> Tuple[Dict[str, Dict[str, object]], Dict[str, float]]:
    """Fit cluster-specific models using the provided training mask."""
    threshold_candidates = list(candidate_thresholds) if candidate_thresholds is not None else None
    cluster_models: Dict[str, Dict[str, object]] = {}
    metrics_buffer: List[Dict[str, float]] = []

    global_majority = int(np.round(bundle.labels[train_mask].mean())) if train_mask.any() else 0
    flags = resolve_pipeline_flags(bundle.name)
    threshold_strategy = flags.get("threshold_strategy", "f1_grid")
    threshold_beta = float(flags.get("threshold_beta", 1.0))
    threshold_min_recall = float(flags.get("threshold_min_recall", 0.0))
    if threshold_strategy == "quantile":
        quantile = float(flags.get("threshold_quantile", 0.5))
        threshold_candidates = [np.clip(quantile, 0.0, 1.0)]

    subset_ratio = float(np.clip(float(flags.get("cluster_train_subset_ratio", 1.0)), 0.0, 1.0))
    label_noise_rate = float(np.clip(float(flags.get("cluster_label_noise", 0.0)), 0.0, 0.49))
    base_seed = int(CONFIG.get("seed", 42))

    def _subset_training_data(
        rng: np.random.RandomState,
        X: np.ndarray,
        y: np.ndarray,
        aux: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        if not (0.0 < subset_ratio < 1.0):
            return X, y, aux
        n_samples = X.shape[0]
        if n_samples <= 2:
            return X, y, aux
        subset_size = int(np.ceil(n_samples * subset_ratio))
        subset_size = min(max(subset_size, 2), n_samples)
        if subset_size >= n_samples:
            return X, y, aux
        indices = rng.choice(n_samples, size=subset_size, replace=False)
        if np.unique(y[indices]).size < 2:
            return X, y, aux
        X_sub = X[indices]
        y_sub = y[indices]
        aux_sub = aux[indices] if aux is not None else None
        return X_sub, y_sub, aux_sub

    def _apply_label_noise(rng: np.random.RandomState, labels: np.ndarray) -> np.ndarray:
        if not (0.0 < label_noise_rate < 0.5):
            return labels
        noisy = labels.copy()
        flips = rng.rand(noisy.size) < label_noise_rate
        if not flips.any():
            return labels
        noisy[flips] = 1 - noisy[flips]
        if np.unique(noisy).size < 2:
            return labels
        return noisy

    # Threshold selection is kept simple: global F-beta grid over raw probabilities.

    global_info: Optional[Dict[str, object]] = None
    raw_global_count = int(train_mask.sum())
    if raw_global_count >= 2 and len(np.unique(bundle.labels[train_mask])) >= 2:
        X_global = bundle.features[train_mask]
        y_global_true = bundle.labels[train_mask]
        global_rng = np.random.RandomState(base_seed + 9973)
        X_global, y_global_true, _ = _subset_training_data(global_rng, X_global, y_global_true)
        y_global = _apply_label_noise(global_rng, y_global_true)

        if len(np.unique(y_global)) >= 2:
            global_class_weights = compute_class_weight(
                class_weight="balanced",
                classes=np.array([0, 1]),
                y=y_global,
            )
            global_weight_map = {0: global_class_weights[0], 1: global_class_weights[1]}
            global_sample_weights = np.vectorize(global_weight_map.get)(y_global)

            global_model = initialize_cluster_model(
                "global",
                int(train_mask.sum()),
                bundle.features.shape[1],
                dataset_name=bundle.name,
                random_state=base_seed,
                feature_names=bundle.feature_names,
                pathway_indices=bundle.pathway_feature_indices,
                drug_indices=bundle.drug_feature_indices,
            )
            global_model.fit(
                X_global,
                y_global,
                sample_weight=global_sample_weights,
            )

            proba_global_train = global_model.predict_proba(X_global)[:, 1]
            threshold_result_global = learn_thresholds(
                threshold_strategy,
                y_global,
                proba_global_train,
                candidate_thresholds=threshold_candidates,
                min_recall=threshold_min_recall,
                beta=threshold_beta,
            )
            global_threshold = float(threshold_result_global.threshold)
            global_pred = (proba_global_train >= global_threshold).astype(int)
            global_metrics = compute_evaluation_metrics(y_global_true, global_pred, proba_global_train)
            global_info = {
                "model": global_model,
                "threshold": global_threshold,
                "trained": True,
                "majority": int(np.round(y_global_true.mean())),
                "train_metrics": global_metrics or {},
                "model_diagnostics": getattr(global_model, "get_training_diagnostics", lambda: {})(),
                "threshold_diagnostics": threshold_result_global.diagnostics,
                "fallback_source": "dataset",
                "train_sample_count": int(y_global_true.shape[0]),
                "raw_train_sample_count": raw_global_count,
                "label_noise_rate": label_noise_rate,
            }

    for cluster_index, cluster_label in enumerate(np.unique(bundle.clusters)):
        key = str(cluster_label)
        cluster_mask = bundle.clusters == cluster_label
        train_cluster_mask = cluster_mask & train_mask
        raw_cluster_count = int(train_cluster_mask.sum())

        info: Dict[str, object] = {
            "model": None,
            "threshold": 0.5,
            "trained": False,
            "majority": global_majority,
        }

        if train_cluster_mask.sum() < 2 or len(np.unique(bundle.labels[train_cluster_mask])) < 2:
            if global_info is not None:
                fallback_info = dict(global_info)
                fallback_info["trained"] = True
                fallback_info["cluster_fallback"] = "dataset"
                fallback_info["majority"] = info["majority"]
                if "train_sample_count" not in fallback_info:
                    fallback_info["train_sample_count"] = raw_cluster_count
                if "raw_train_sample_count" not in fallback_info:
                    fallback_info["raw_train_sample_count"] = raw_cluster_count
                if "label_noise_rate" not in fallback_info:
                    fallback_info["label_noise_rate"] = label_noise_rate
                cluster_models[key] = fallback_info
                continue
            info["train_sample_count"] = raw_cluster_count
            info["raw_train_sample_count"] = raw_cluster_count
            info["label_noise_rate"] = label_noise_rate
            cluster_models[key] = info
            continue

        X_train = bundle.features[train_cluster_mask]
        y_train_true = bundle.labels[train_cluster_mask]
        cluster_labels_all = bundle.clusters[train_cluster_mask]

        cluster_rng = np.random.RandomState(base_seed + 100 + cluster_index)
        X_train, y_train_true, cluster_labels_all = _subset_training_data(
            cluster_rng,
            X_train,
            y_train_true,
            cluster_labels_all,
        )
        y_train = _apply_label_noise(cluster_rng, y_train_true)
        majority_true = int(np.round(y_train_true.mean())) if y_train_true.size else info["majority"]

        if len(np.unique(y_train)) < 2:
            if global_info is not None:
                fallback_info = dict(global_info)
                fallback_info["trained"] = True
                fallback_info["cluster_fallback"] = "dataset"
                fallback_info["majority"] = majority_true
                if "train_sample_count" not in fallback_info:
                    fallback_info["train_sample_count"] = int(y_train_true.shape[0])
                if "raw_train_sample_count" not in fallback_info:
                    fallback_info["raw_train_sample_count"] = raw_cluster_count
                if "label_noise_rate" not in fallback_info:
                    fallback_info["label_noise_rate"] = label_noise_rate
                cluster_models[key] = fallback_info
            else:
                info["train_sample_count"] = int(y_train_true.shape[0])
                info["raw_train_sample_count"] = raw_cluster_count
                info["label_noise_rate"] = label_noise_rate
                info["majority"] = majority_true
                cluster_models[key] = info
            continue

        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.array([0, 1]),
            y=y_train,
        )
        weight_map = {0: class_weights[0], 1: class_weights[1]}
        sample_weights = np.vectorize(weight_map.get)(y_train)

        model = initialize_cluster_model(
            key,
            raw_cluster_count,
            bundle.features.shape[1],
            dataset_name=bundle.name,
            random_state=base_seed + cluster_index,
            feature_names=bundle.feature_names,
            pathway_indices=bundle.pathway_feature_indices,
            drug_indices=bundle.drug_feature_indices,
        )
        fit_kwargs: Dict[str, object] = {
            "sample_weight": sample_weights,
            "cluster_labels": cluster_labels_all,
        }

        if bundle.prototypes:
            cluster_proto_map = bundle.prototypes.get("clusters", {})
            vectors: List[np.ndarray] = []
            labels: List[str] = []

            current_entry = cluster_proto_map.get(key)
            if current_entry and "all" in current_entry:
                vectors.append(np.asarray(current_entry["all"], dtype=float).reshape(1, -1))
                labels.append(f"{key}_self")

            for other_key, entry in cluster_proto_map.items():
                if other_key == key or "all" not in entry:
                    continue
                vectors.append(np.asarray(entry["all"], dtype=float).reshape(1, -1))
                labels.append(f"{other_key}_cluster")

            global_proto = bundle.prototypes.get("global", {}).get("all")
            if global_proto is not None:
                vectors.append(np.asarray(global_proto, dtype=float).reshape(1, -1))
                labels.append("global_all")

            if vectors:
                stacked = np.vstack(vectors)
                fit_kwargs["initial_prototypes"] = stacked
                fit_kwargs["prototype_labels"] = labels

        model.fit(
            X_train,
            y_train,
            **fit_kwargs,
        )

        proba_train = model.predict_proba(X_train)[:, 1]
        threshold_result = learn_thresholds(
            threshold_strategy,
            y_train,
            proba_train,
            candidate_thresholds=threshold_candidates,
            min_recall=threshold_min_recall,
            beta=threshold_beta,
        )
        best_threshold = float(threshold_result.threshold)
        train_pred = (proba_train >= best_threshold).astype(int)
        best_metrics = compute_evaluation_metrics(y_train_true, train_pred, proba_train)

        model_diagnostics = getattr(model, "get_training_diagnostics", lambda: {})()
        if isinstance(model_diagnostics, dict):
            model_diagnostics["feature_names"] = list(bundle.feature_names)
            model_diagnostics["pathway_feature_indices"] = list(bundle.pathway_feature_indices or [])
            model_diagnostics["drug_feature_indices"] = list(bundle.drug_feature_indices or [])

        info.update(
            {
                "model": model,
                "threshold": best_threshold,
                "trained": True,
                "majority": majority_true,
                "train_metrics": best_metrics or {},
                "model_diagnostics": model_diagnostics,
                "threshold_diagnostics": threshold_result.diagnostics,
                "train_sample_count": int(y_train_true.shape[0]),
                "raw_train_sample_count": raw_cluster_count,
                "label_noise_rate": label_noise_rate,
            }
        )
        cluster_models[key] = info
        if best_metrics:
            metrics_buffer.append(best_metrics)

    summary = aggregate_metric_list(metrics_buffer)
    return cluster_models, summary


def predict_with_models_on_mask(
    bundle: DatasetBundle,
    cluster_models: Mapping[str, Dict[str, object]],
    mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate predictions for the subset specified by ``mask``."""
    y_pred_full = np.zeros_like(bundle.labels)
    y_prob_full = np.zeros_like(bundle.labels, dtype=float)

    global_majority = int(np.round(bundle.labels[mask].mean())) if mask.any() else 0

    for cluster_label in np.unique(bundle.clusters):
        key = str(cluster_label)
        cluster_mask = (bundle.clusters == cluster_label) & mask
        if not cluster_mask.any():
            continue

        info = cluster_models.get(key)
        if info and info.get("model") is not None:
            model = info["model"]
            threshold = info.get("threshold", 0.5)
            proba = model.predict_proba(bundle.features[cluster_mask])[:, 1]
            preds = (proba >= threshold).astype(int)
        else:
            majority = int(np.round(info.get("majority", global_majority))) if info else global_majority
            proba = np.full(cluster_mask.sum(), majority, dtype=float)
            preds = np.full(cluster_mask.sum(), majority, dtype=int)

        y_pred_full[cluster_mask] = preds
        y_prob_full[cluster_mask] = proba

    return y_pred_full[mask], y_prob_full[mask]


def _fbeta_from_scores(precision: float, recall: float, beta: float) -> float:
    beta_sq = float(beta) ** 2
    if (precision == 0.0 and recall == 0.0) or beta_sq == float("inf"):
        return 0.0
    denominator = beta_sq * precision + recall
    if denominator == 0.0:
        return 0.0
    return (1.0 + beta_sq) * precision * recall / denominator


def create_train_test_split(
    bundle: DatasetBundle,
    seed: int = 42,
    test_size: float = 0.25,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create group-aware train/test masks."""
    rng = np.random.RandomState(seed)
    n_samples = bundle.labels.shape[0]

    groups = bundle.groups
    labels = bundle.labels
    indices = np.arange(n_samples)

    if groups is None:
        groups = np.arange(n_samples, dtype=int)

    def _stratified_holdout():
        splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=min(max(test_size, 1 / max(n_samples, 1)), 0.5),
            random_state=seed,
        )
        return next(splitter.split(indices, labels))

    train_idx = test_idx = None
    unique_groups = np.unique(groups)
    if len(unique_groups) > 1:
        for attempt in range(5):
            splitter = GroupShuffleSplit(
                n_splits=1,
                test_size=min(max(test_size, 1 / len(unique_groups)), 0.5),
                random_state=seed + attempt,
            )
            candidate_train, candidate_test = next(splitter.split(indices, labels, groups))
            if (
                np.unique(labels[candidate_test]).size >= 2
                and np.unique(labels[candidate_train]).size >= 2
            ):
                train_idx, test_idx = candidate_train, candidate_test
                break

    if train_idx is None or test_idx is None:
        logger.debug(
            "{} train/test split falling back to stratified shuffle.",
            bundle.name,
        )
        if np.unique(labels).size < 2:
            train_idx = indices
            test_idx = np.array([], dtype=int)
        else:
            train_idx, test_idx = _stratified_holdout()

    if np.unique(labels[test_idx]).size < 2 and np.unique(labels).size >= 2:
        minority_label = 0 if (labels == 0).sum() <= (labels == 1).sum() else 1
        minority_indices = np.where(labels == minority_label)[0]
        available = np.setdiff1d(minority_indices, test_idx, assume_unique=False)
        if available.size > 0:
            test_idx = np.append(test_idx, available[0])
            train_idx = train_idx[train_idx != available[0]]

    if test_idx.size == 0:
        test_idx = np.array([train_idx[-1]])
        train_idx = train_idx[:-1]

    train_mask = np.zeros(n_samples, dtype=bool)
    test_mask = np.zeros(n_samples, dtype=bool)
    train_mask[train_idx] = True
    test_mask[test_idx] = True

    logger.debug(
        "{} train/test split -> train groups: {}, test groups: {}",
        bundle.name,
        len(np.unique(groups[train_mask])),
        len(np.unique(groups[test_mask])),
    )
    return train_mask, test_mask


def cross_validate_dataset(
    bundle: DatasetBundle,
    train_mask: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 4,
    seed: int = 42,
    candidate_thresholds: Optional[Iterable[float]] = None,
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    """Run group-aware cross-validation on the training subset."""
    train_indices = np.where(train_mask)[0]
    if train_indices.size == 0:
        return {"n_folds": 0}, []

    labels = bundle.labels[train_indices]
    train_groups = groups[train_indices]
    unique_groups = np.unique(train_groups)

    if len(unique_groups) >= n_splits:
        splitter = GroupKFold(n_splits=n_splits)
        splits_iter = list(splitter.split(train_indices, labels, train_groups))
        effective_splits = n_splits
    else:
        effective_splits = min(n_splits, max(2, train_indices.size))
        splitter = StratifiedKFold(n_splits=effective_splits, shuffle=True, random_state=seed)
        try:
            splits_iter = list(splitter.split(train_indices, labels))
        except ValueError:
            rng = np.random.RandomState(seed)
            rel_indices = np.arange(train_indices.size)
            rng.shuffle(rel_indices)
            split_point = max(1, int(rel_indices.size * 0.8))
            splits_iter = [(rel_indices[:split_point], rel_indices[split_point:])]
            effective_splits = len(splits_iter)

    def _has_diverse_folds(splits):
        for _, val_rel in splits:
            if val_rel.size == 0:
                continue
            if len(np.unique(labels[val_rel])) >= 2:
                return True
        return False

    if not _has_diverse_folds(splits_iter):
        label_counts = np.bincount(labels)
        valid_counts = label_counts[label_counts > 0]
        if valid_counts.size == 0 or valid_counts.min() < 2:
            logger.warning(
                "{}: insufficient class diversity to perform cross-validation.",
                bundle.name,
            )
            return {"n_folds": 0}, []
        new_n_splits = min(n_splits, int(valid_counts.min()))
        splitter = StratifiedKFold(n_splits=new_n_splits, shuffle=True, random_state=seed)
        splits_iter = list(splitter.split(train_indices, labels))
        effective_splits = new_n_splits
        if not _has_diverse_folds(splits_iter):
            logger.warning(
                "{}: stratified fallback still produced single-class validation folds; skipping CV.",
                bundle.name,
            )
            return {"n_folds": 0}, []

    fold_metrics: List[Dict[str, float]] = []

    for train_rel, val_rel in splits_iter:
        if val_rel.size == 0:
            continue
        fold_train_idx = train_indices[train_rel]
        val_idx = train_indices[val_rel]

        fold_train_mask = np.zeros_like(bundle.labels, dtype=bool)
        val_mask = np.zeros_like(bundle.labels, dtype=bool)
        fold_train_mask[fold_train_idx] = True
        val_mask[val_idx] = True

        if len(np.unique(bundle.labels[val_mask])) < 2:
            continue

        cluster_models, _ = train_cluster_models_single(
            bundle,
            fold_train_mask,
            candidate_thresholds=candidate_thresholds,
        )
        y_pred, y_prob = predict_with_models_on_mask(bundle, cluster_models, val_mask)
        metrics = compute_evaluation_metrics(bundle.labels[val_mask], y_pred, y_prob)
        fold_metrics.append(metrics)

    if not fold_metrics:
        logger.warning(
            "{}: cross-validation skipped due to insufficient class diversity in validation folds.",
            bundle.name,
        )
        return {"n_folds": 0}, []

    summary = aggregate_metric_list(fold_metrics)
    summary["n_folds"] = effective_splits
    return summary, fold_metrics


def train_full_models(
    balanced_datasets: Mapping[str, DatasetBundle],
    train_masks: Mapping[str, np.ndarray],
    candidate_thresholds: Optional[Iterable[float]] = None,
) -> Dict[str, Dict[str, Dict[str, object]]]:
    """Train final models on the full dataset for interpretability outputs."""
    final_models: Dict[str, Dict[str, Dict[str, object]]] = {}
    for dataset_name, bundle in balanced_datasets.items():
        train_mask = train_masks.get(dataset_name)
        if train_mask is None or not train_mask.any():
            continue
        models, _ = train_cluster_models_single(
            bundle,
            train_mask,
            candidate_thresholds=candidate_thresholds,
        )
        final_models[dataset_name] = models
    return final_models


def interpretability_analysis(
    optimised_models: Mapping[str, Dict[str, Dict[str, object]]],
    balanced_datasets: Mapping[str, DatasetBundle],
    top_k: int = 10,
) -> Dict[str, Dict[str, List[Dict[str, float]]]]:
    """Extract model feature importances."""
    results: Dict[str, Dict[str, List[Dict[str, float]]]] = {}

    for dataset_name, models in optimised_models.items():
        bundle = balanced_datasets[dataset_name]
        dataset_results: Dict[str, List[Dict[str, float]]] = {}

        for cluster_id, model_info in models.items():
            model = model_info.get("model")
            if model is None:
                continue
            if not hasattr(model, "get_feature_importance"):
                continue
            summary = model.get_feature_importance(bundle.feature_names, top_k=top_k)
            if summary:
                dataset_results[cluster_id] = summary

        results[dataset_name] = dataset_results

    return results


def comprehensive_results_and_visualization(
    final_models: Mapping[str, Dict[str, Dict[str, object]]],
    balanced_datasets: Mapping[str, DatasetBundle],
    interpretability_results: Mapping[str, Dict[str, List[Dict[str, float]]]],
    evaluation_results: Mapping[str, Dict[str, object]],
    test_results: Mapping[str, Dict[str, float]],
    results_manager,
) -> Dict[str, Dict[str, object]]:
    """Persist metrics and collate a compact summary for downstream use."""
    summary: Dict[str, Dict[str, object]] = {}

    for dataset_name, evaluation in evaluation_results.items():
        bundle = balanced_datasets[dataset_name]
        metrics = evaluation["metrics"]
        cluster_thresholds = [
            info.get("threshold", 0.5) for info in final_models.get(dataset_name, {}).values()
        ]
        avg_threshold = float(np.mean(cluster_thresholds)) if cluster_thresholds else 0.5

        mean_auc = metrics.get("mean_auc", float("nan"))
        mean_pr_auc = metrics.get("mean_pr_auc", float("nan"))
        mean_acc = metrics.get("mean_accuracy", float("nan"))
        mean_prec = metrics.get("mean_precision", float("nan"))
        mean_rec = metrics.get("mean_recall", float("nan"))
        mean_f1 = metrics.get("mean_f1", float("nan"))

        results_manager.append_log(
            f"{dataset_name}: "
            f"AUC={mean_auc:.3f} (CI95 {metrics.get('ci_auc_low', float('nan')):.3f}, "
            f"{metrics.get('ci_auc_high', float('nan')):.3f}), "
            f"PR_AUC={mean_pr_auc:.3f} (CI95 {metrics.get('ci_pr_auc_low', float('nan')):.3f}, "
            f"{metrics.get('ci_pr_auc_high', float('nan')):.3f}), "
            f"ACC={mean_acc:.3f} (CI95 {metrics.get('ci_accuracy_low', float('nan')):.3f}, "
            f"{metrics.get('ci_accuracy_high', float('nan')):.3f}), "
            f"PREC={mean_prec:.3f} (CI95 {metrics.get('ci_precision_low', float('nan')):.3f}, "
            f"{metrics.get('ci_precision_high', float('nan')):.3f}), "
            f"REC={mean_rec:.3f} (CI95 {metrics.get('ci_recall_low', float('nan')):.3f}, "
            f"{metrics.get('ci_recall_high', float('nan')):.3f}), "
            f"F1={mean_f1:.3f} (CI95 {metrics.get('ci_f1_low', float('nan')):.3f}, "
            f"{metrics.get('ci_f1_high', float('nan')):.3f}), "
            f"threshold~{avg_threshold:.2f}",
        )

        results_manager.append_row(
            {
                "split": "cv",
                "dataset": dataset_name,
                "drug": bundle.metadata.get("drug_name", "unknown"),
                "n_target": int(bundle.features.shape[0]),
                "n_clusters": int(len(np.unique(bundle.clusters))),
                "n_features": int(bundle.features.shape[1]),
                "mean_auc": mean_auc,
                "std_auc": float(metrics.get("std_auc", float("nan"))),
                "ci_auc_low": float(metrics.get("ci_auc_low", float("nan"))),
                "ci_auc_high": float(metrics.get("ci_auc_high", float("nan"))),
                "mean_accuracy": mean_acc,
                "std_accuracy": float(metrics.get("std_accuracy", float("nan"))),
                "ci_accuracy_low": float(metrics.get("ci_accuracy_low", float("nan"))),
                "ci_accuracy_high": float(metrics.get("ci_accuracy_high", float("nan"))),
                "mean_precision": mean_prec,
                "std_precision": float(metrics.get("std_precision", float("nan"))),
                "ci_precision_low": float(metrics.get("ci_precision_low", float("nan"))),
                "ci_precision_high": float(metrics.get("ci_precision_high", float("nan"))),
                "mean_recall": mean_rec,
                "std_recall": float(metrics.get("std_recall", float("nan"))),
                "ci_recall_low": float(metrics.get("ci_recall_low", float("nan"))),
                "ci_recall_high": float(metrics.get("ci_recall_high", float("nan"))),
                "mean_f1": mean_f1,
                "std_f1": float(metrics.get("std_f1", float("nan"))),
                "ci_f1_low": float(metrics.get("ci_f1_low", float("nan"))),
                "ci_f1_high": float(metrics.get("ci_f1_high", float("nan"))),
                "mean_pr_auc": mean_pr_auc,
                "std_pr_auc": float(metrics.get("std_pr_auc", float("nan"))),
                "ci_pr_auc_low": float(metrics.get("ci_pr_auc_low", float("nan"))),
                "ci_pr_auc_high": float(metrics.get("ci_pr_auc_high", float("nan"))),
                "threshold": avg_threshold,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }
        )

        summary[dataset_name] = {
            "metrics": metrics,
            "n_samples": int(bundle.features.shape[0]),
            "n_clusters": int(len(np.unique(bundle.clusters))),
            "interpretability": interpretability_results.get(dataset_name, {}),
        }
        if dataset_name in test_results:
            summary[dataset_name]["test_metrics"] = test_results[dataset_name]

    return summary


# --------------------------------------------------------------------------- #
#                                   Runner                                    #
# --------------------------------------------------------------------------- #


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the scRADAR pipeline."
    )
    parser.add_argument(
        "--config",
        default="config.py",
        help="Configuration module (informational; CONFIG is imported at module load).",
    )
    parser.add_argument(
        "--run-name",
        default=os.environ.get("RUN_NAME", "run"),
        help="Subdirectory under results/log roots for this run.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Entry point mirroring the behaviour of the baseline scripts."""
    args = parse_args(argv)
    run_name = str(args.run_name or "run").strip() or "run"
    np.random.seed(CONFIG.get("seed", 42))

    # ---- 根据 config 里声明的 dataset_names 子集，筛选实际要跑的 datasets_config ----
    all_datasets_config = CONFIG["datasets_config"]
    selected_names = CONFIG.get("dataset_names")

    if selected_names:
        # 只保留在 datasets_config 中存在的名字（TRAIN_DATASETS / ONLY_DATASET）
        datasets_config = {
            name: all_datasets_config[name]
            for name in selected_names
            if name in all_datasets_config
        }
    else:
        # 如果没显式指定，就退回到全量（一般用不到）
        datasets_config = all_datasets_config

    if not datasets_config:
        raise RuntimeError(
            "No valid datasets selected. "
            "Check CONFIG['dataset_names'] or ONLY_DATASET environment variable."
        )

    # 之后所有步骤只围绕 datasets_config 这一小撮数据集
    log_dir = setup_logging(datasets_config, run_name=run_name)
    logger.info("Logging configured under {}", log_dir)

    results_manager = get_results_manager(
        run_name,
        config=CONFIG,
        fieldnames=[
            "split",
            "dataset",
            "drug",
            "n_target",
            "n_clusters",
            "n_features",
            "mean_auc",
            "std_auc",
            "ci_auc_low",
            "ci_auc_high",
            "mean_pr_auc",
            "std_pr_auc",
            "ci_pr_auc_low",
            "ci_pr_auc_high",
            "mean_accuracy",
            "std_accuracy",
            "ci_accuracy_low",
            "ci_accuracy_high",
            "mean_precision",
            "std_precision",
            "ci_precision_low",
            "ci_precision_high",
            "mean_recall",
            "std_recall",
            "ci_recall_low",
            "ci_recall_high",
            "mean_f1",
            "std_f1",
            "ci_f1_low",
            "ci_f1_high",
            "threshold",
            "timestamp",
        ],
    )
    seed = CONFIG.get("seed", 42)
    results_manager.append_log(f"Run started (seed={seed}).")

    raw_datasets = load_multi_datasets(datasets_config)
    drug_fingerprints: Dict[str, Dict[str, np.ndarray]] = {}

    standardized_datasets, _ = standardize_multi_dataset_responses(raw_datasets)
    cleaned_datasets, _ = preprocess_multi_datasets(standardized_datasets)
    engineered_datasets, _ = enhanced_feature_engineering_multi(cleaned_datasets)
    clustered_datasets, _ = cell_clustering_and_subgroups(engineered_datasets)
    joint_datasets, _ = construct_cell_drug_joint_features(clustered_datasets, drug_fingerprints)
    balanced_datasets, _ = handle_cluster_aware_imbalance(joint_datasets)

    evaluation_results: Dict[str, Dict[str, object]] = {}
    train_masks: Dict[str, np.ndarray] = {}
    test_masks: Dict[str, np.ndarray] = {}
    test_results: Dict[str, Dict[str, float]] = {}
    total = 0
    successes = 0

    metric_order = ["auc", "pr_auc", "accuracy", "precision", "recall", "f1"]

    for dataset_name in datasets_config.keys():
        bundle = balanced_datasets.get(dataset_name)
        if bundle is None:
            continue

        total += 1
        logger.info("=== Dataset: {} ===", dataset_name)
        results_manager.append_log(f"=== Dataset: {dataset_name} ===")

        metadata = bundle.metadata
        source_value = metadata.get("n_source", "NA")

        parts = [
            f"{dataset_name}: drug={metadata.get('drug_name', 'unknown')}",
            f"source={source_value}",
            f"target={metadata.get('n_target', bundle.features.shape[0])}",
            f"genes={metadata.get('genes', bundle.features.shape[1])}",
            f"strategy={metadata.get('label_strategy', 'unknown')}",
        ]
        dataset_line = " | ".join(parts)
        logger.info(dataset_line)
        results_manager.append_log(dataset_line)

        train_mask, test_mask = create_train_test_split(bundle, seed=seed)
        train_masks[dataset_name] = train_mask
        test_masks[dataset_name] = test_mask

        groups = bundle.groups if bundle.groups is not None else np.arange(bundle.labels.shape[0])

        summary, fold_metrics = cross_validate_dataset(
            bundle,
            train_mask=train_mask,
            groups=groups,
            n_splits=4,
            seed=seed,
        )

        if fold_metrics:
            for fold_idx, metrics in enumerate(fold_metrics, start=1):
                fold_line = (
                    f"{dataset_name} fold {fold_idx}: "
                    + ", ".join(
                        f"{metric.upper()}={metrics.get(metric, float('nan')):.3f}"
                        for metric in metric_order
                    )
                )
                logger.info(fold_line)
                results_manager.append_log(fold_line)

            summary_line = (
                f"{dataset_name} summary: "
                + ", ".join(
                    f"{metric.upper()}={summary.get(f'mean_{metric}', float('nan')):.3f} "
                    f"(CI95 {summary.get(f'ci_{metric}_low', float('nan')):.3f}, "
                    f"{summary.get(f'ci_{metric}_high', float('nan')):.3f})"
                    for metric in metric_order
                )
            )
            logger.info(summary_line)
            results_manager.append_log(summary_line)
        else:
            skip_line = (
                f"{dataset_name} cross-validation skipped due to insufficient class diversity."
            )
            logger.warning(skip_line)
            results_manager.append_log(skip_line)

        evaluation_results[dataset_name] = {
            "metrics": summary,
            "fold_metrics": fold_metrics,
        }

        if not np.isnan(summary.get("mean_accuracy", float("nan"))):
            successes += 1

    final_models = train_full_models(
        balanced_datasets,
        train_masks,
    )

    for dataset_name, bundle in balanced_datasets.items():
        if dataset_name not in datasets_config:
            continue

        models = final_models.get(dataset_name)
        test_mask = test_masks.get(dataset_name)
        if models is None or test_mask is None or not test_mask.any():
            logger.warning(
                "{} skipped test evaluation due to empty test set.",
                dataset_name,
            )
            continue

        flags = resolve_pipeline_flags(dataset_name)
        y_pred, y_prob = predict_with_models_on_mask(
            bundle,
            models,
            test_mask,
        )
        metrics = compute_evaluation_metrics(bundle.labels[test_mask], y_pred, y_prob)
        boot_ci = bootstrap_evaluation_ci(bundle.labels[test_mask], y_pred, y_prob)
        test_results[dataset_name] = metrics
        test_line = (
            f"{dataset_name} test: "
            + ", ".join(
                f"{metric.upper()}={metrics.get(metric, float('nan')):.3f}"
                for metric in metric_order
            )
        )
        logger.info(test_line)
        results_manager.append_log(test_line)
        n_pos = int(bundle.labels[test_mask].sum())
        n_neg = int(test_mask.sum() - n_pos)
        model_type = str(flags.get("model_type", "proto_mech"))
        thresholds = [
            info.get("threshold", 0.5)
            for info in models.values()
            if info.get("trained")
        ]
        avg_threshold = float(np.mean(thresholds)) if thresholds else float("nan")
        results_manager.append_row(
            {
                "split": "test",
                "dataset": dataset_name,
                "drug": bundle.metadata.get("drug_name", "unknown"),
                "n_target": int(bundle.features.shape[0]),
                "n_clusters": int(len(np.unique(bundle.clusters))),
                "n_features": int(bundle.features.shape[1]),
                "mean_auc": float(metrics.get("auc", float("nan"))),
                "std_auc": float(boot_ci.get("std_auc", float("nan"))),
                "ci_auc_low": float(boot_ci.get("ci_auc_low", float("nan"))),
                "ci_auc_high": float(boot_ci.get("ci_auc_high", float("nan"))),
                "mean_pr_auc": float(metrics.get("pr_auc", float("nan"))),
                "std_pr_auc": float(boot_ci.get("std_pr_auc", float("nan"))),
                "ci_pr_auc_low": float(boot_ci.get("ci_pr_auc_low", float("nan"))),
                "ci_pr_auc_high": float(boot_ci.get("ci_pr_auc_high", float("nan"))),
                "mean_accuracy": float(metrics.get("accuracy", float("nan"))),
                "std_accuracy": float(boot_ci.get("std_accuracy", float("nan"))),
                "ci_accuracy_low": float(boot_ci.get("ci_accuracy_low", float("nan"))),
                "ci_accuracy_high": float(boot_ci.get("ci_accuracy_high", float("nan"))),
                "mean_precision": float(metrics.get("precision", float("nan"))),
                "std_precision": float(boot_ci.get("std_precision", float("nan"))),
                "ci_precision_low": float(boot_ci.get("ci_precision_low", float("nan"))),
                "ci_precision_high": float(boot_ci.get("ci_precision_high", float("nan"))),
                "mean_recall": float(metrics.get("recall", float("nan"))),
                "std_recall": float(boot_ci.get("std_recall", float("nan"))),
                "ci_recall_low": float(boot_ci.get("ci_recall_low", float("nan"))),
                "ci_recall_high": float(boot_ci.get("ci_recall_high", float("nan"))),
                "mean_f1": float(metrics.get("f1", float("nan"))),
                "std_f1": float(boot_ci.get("std_f1", float("nan"))),
                "ci_f1_low": float(boot_ci.get("ci_f1_low", float("nan"))),
                "ci_f1_high": float(boot_ci.get("ci_f1_high", float("nan"))),
                "threshold": avg_threshold,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }
        )
        append_perf_artifact(
            dataset_name,
            "test",
            metrics,
            n_pos,
            n_neg,
            model_type,
            bundle,
            results_manager,
            threshold_used=avg_threshold,
        )

    interpretability_results = interpretability_analysis(final_models, balanced_datasets)
    summary_payload = comprehensive_results_and_visualization(
        final_models,
        balanced_datasets,
        interpretability_results,
        evaluation_results,
        test_results,
        results_manager,
    )
    # 写 summary.json 和 prototype_diagnostics.json
    write_summary_and_prototypes(
        summary_payload,
        final_models,
        balanced_datasets,
        results_manager,
    )

    # 写模型 checkpoint
    save_model_checkpoint(final_models, results_manager)

    # 结束日志（注意 successes / total 在 main 作用域里是有的）
    logger.info("Run completed: {}/{} datasets processed.", successes, total)
    results_manager.append_log(f"Run completed: {successes}/{total} datasets processed.")


def _artifact_dir(results_manager, dataset_name: Optional[str] = None) -> Path:
    base = Path(results_manager.base_dir) / "artifacts"
    if dataset_name:
        base = base / str(dataset_name)
    base.mkdir(parents=True, exist_ok=True)
    return base


def append_perf_artifact(
    dataset_name: str,
    subset_name: str,
    metrics: Mapping[str, float],
    n_pos: int,
    n_neg: int,
    model_type: str,
    bundle: DatasetBundle,
    results_manager,
    threshold_used: Optional[float] = None,
) -> None:
    path = _artifact_dir(results_manager) / "perf_by_fold.csv"
    fieldnames = [
        "dataset",
        "subset",
        "drug",
        "model_type",
        "n_pos",
        "n_neg",
        "auc",
        "pr_auc",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "threshold",
        "timestamp",
    ]
    write_header = not path.exists()
    row = {
        "dataset": dataset_name,
        "subset": subset_name,
        "drug": bundle.metadata.get("drug_name", "unknown"),
        "model_type": model_type,
        "n_pos": int(n_pos),
        "n_neg": int(n_neg),
        "auc": float(metrics.get("auc", float("nan"))),
        "pr_auc": float(metrics.get("pr_auc", float("nan"))),
        "accuracy": float(metrics.get("accuracy", float("nan"))),
        "precision": float(metrics.get("precision", float("nan"))),
        "recall": float(metrics.get("recall", float("nan"))),
        "f1": float(metrics.get("f1", float("nan"))),
        "threshold": float(threshold_used) if threshold_used is not None else float("nan"),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def write_summary_and_prototypes(
    summary_payload: Dict[str, Dict[str, object]],
    final_models: Mapping[str, Dict[str, Dict[str, object]]],
    balanced_datasets: Mapping[str, DatasetBundle],
    results_manager,
) -> None:
    base_dir = Path(results_manager.base_dir)

    # --------- summary.json ----------
    summary_path = base_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2, ensure_ascii=False)
    logger.info("Summary JSON saved to {}", summary_path)

    # --------- prototype_diagnostics.json ----------
    proto_payload: Dict[str, object] = {}

    for dataset_name, models in final_models.items():
        bundle = balanced_datasets[dataset_name]
        clusters_entry: Dict[str, object] = {}

        for cluster_id, info in models.items():
            diag = info.get("model_diagnostics") or {}
            clusters_entry[str(cluster_id)] = {
                "cluster_id": str(cluster_id),
                "train_sample_count": info.get("train_sample_count"),
                "raw_train_sample_count": info.get("raw_train_sample_count", info.get("train_sample_count")),
                "label_noise_rate": info.get("label_noise_rate"),
                "threshold": info.get("threshold"),
                **diag,
            }

        proto_payload[dataset_name] = {
            "n_clusters": len(models),
            "feature_dim": int(bundle.features.shape[1]),
            "clusters": clusters_entry,
        }
        mechanism_profile = {}
        if bundle.metadata.get("drug_pathway_profile"):
            mechanism_profile["pathway_profile"] = bundle.metadata["drug_pathway_profile"]
        if bundle.metadata.get("drug_static_tokens"):
            mechanism_profile["static_tokens"] = bundle.metadata["drug_static_tokens"]
        if mechanism_profile:
            proto_payload[dataset_name]["drug_mechanism_profile"] = mechanism_profile
        cluster_match = bundle.metadata.get("cluster_mechanism_scores")
        if cluster_match:
            proto_payload[dataset_name]["cluster_mechanism_match"] = cluster_match

    proto_path = base_dir / "prototype_diagnostics.json"
    with proto_path.open("w", encoding="utf-8") as f:
        json.dump(proto_payload, f, indent=2, ensure_ascii=False)
    logger.info("Prototype diagnostics saved to {}", proto_path)


def _to_numpy_dict(state_dict) -> Dict[str, np.ndarray]:
    converted: Dict[str, np.ndarray] = {}
    for key, value in state_dict.items():
        if hasattr(value, "detach"):
            converted[key] = value.detach().cpu().numpy()
        elif hasattr(value, "cpu"):
            converted[key] = value.cpu().numpy()
        else:
            converted[key] = np.asarray(value)
    return converted


def _serialize_model_state(model: object) -> Dict[str, object]:
    try:
        import torch
    except ImportError:  # pragma: no cover - defensive
        torch = None

    if hasattr(model, "_prototypes") and hasattr(model, "_head") and torch is not None:
        state_np: Dict[str, np.ndarray] = {}
        prototypes = getattr(model, "_prototypes", None)
        if prototypes is not None:
            if hasattr(prototypes, "detach"):
                state_np["prototypes"] = prototypes.detach().cpu().numpy()
            else:
                state_np["prototypes"] = np.asarray(prototypes)
        head = getattr(model, "_head", None)
        if head is not None and hasattr(head, "state_dict"):
            head_state = head.state_dict()
            head_np = _to_numpy_dict(head_state)
            for key, value in head_np.items():
                state_np[f"head.{key}"] = value
        film = getattr(model, "_film", None)
        if film is not None and hasattr(film, "state_dict"):
            film_state = film.state_dict()
            film_np = _to_numpy_dict(film_state)
            for key, value in film_np.items():
                state_np[f"film.{key}"] = value
        return {
            "type": "proto_mech",
            "state_dict": state_np,
            "num_prototypes": getattr(model, "num_prototypes", None),
            "top_k": getattr(model, "top_k", None),
            "temperature": getattr(model, "temperature", None),
            "dropout": getattr(model, "dropout", None),
        }

    return {"type": type(model).__name__}


def save_model_checkpoint(
    final_models: Mapping[str, Dict[str, Dict[str, object]]],
    results_manager,
) -> None:
    """Persist trained models for external evaluation."""

    payload: Dict[str, object] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "models": {},
        "version": "1.0",
        "pipeline_flags": CONFIG.get("pipeline_flags", {}),
    }

    for dataset_name, models in final_models.items():
        dataset_entry: Dict[str, object] = {}
        for cluster_id, info in models.items():
            model = info.get("model")
            if model is None:
                continue
            dataset_entry[cluster_id] = {
                "model": _serialize_model_state(model),
                "threshold": info.get("threshold"),
                "threshold_diagnostics": info.get("threshold_diagnostics"),
                "model_diagnostics": info.get("model_diagnostics"),
            }
        payload["models"][dataset_name] = dataset_entry

    checkpoint_path = Path(results_manager.base_dir) / "model_checkpoint.npz"
    np.savez_compressed(checkpoint_path, payload=np.array(payload, dtype=object))
    logger.info("Checkpoint saved at {}", checkpoint_path)


if __name__ == "__main__":
    main()
