"""Feature construction helpers for gene and pathway representations."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from loguru import logger

try:
    from scipy import sparse
except ImportError:  # pragma: no cover - optional dependency
    sparse = None

import scanpy as sc  # type: ignore

_FALLBACK_PATHWAY_NOTICE: set[str] = set()

@dataclass(frozen=True)
class PathwayDefinition:
    """Internal representation of a pathway mapped to dataset indices."""

    name: str
    indices: np.ndarray
    complement: np.ndarray
    weights: Optional[np.ndarray] = None


def _to_dense(matrix) -> np.ndarray:
    if sparse is not None and sparse.issparse(matrix):
        return matrix.toarray()
    return np.asarray(matrix)


def _ensure_pca(adata: "sc.AnnData", max_components: int = 50) -> np.ndarray:
    """Ensure ``adata`` has a PCA embedding and return it."""
    if "X_pca" not in adata.obsm:
        n_components = min(max_components, max(2, adata.n_vars - 1))
        sc.tl.pca(adata, n_comps=n_components, svd_solver="arpack")
    return np.asarray(adata.obsm["X_pca"], dtype=float)


def compute_gene_only_features(
    adata: "sc.AnnData",
    *,
    max_components: int = 50,
) -> Tuple[np.ndarray, list[str], Dict[str, float]]:
    """Return PCA-based representations using only gene expression."""
    pca_matrix = _ensure_pca(adata, max_components=max_components)
    feature_names = [f"pca_{idx + 1}" for idx in range(pca_matrix.shape[1])]
    diagnostics = {
        "mode": "gene_only",
        "n_gene_features": int(pca_matrix.shape[1]),
    }
    return pca_matrix, feature_names, diagnostics


def _rank_transform(matrix: np.ndarray) -> np.ndarray:
    order = np.argsort(matrix, axis=1, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    rows = np.arange(matrix.shape[0])[:, None]
    ranks[rows, order] = np.arange(1, matrix.shape[1] + 1, dtype=float)
    ranks /= float(matrix.shape[1])
    return ranks


def _zscore(matrix: np.ndarray) -> np.ndarray:
    mean = np.mean(matrix, axis=1, keepdims=True)
    std = np.std(matrix, axis=1, keepdims=True)
    std = np.where(std == 0, 1.0, std)
    return (matrix - mean) / std


def _parse_gmt(path: Path) -> Mapping[str, List[str]]:
    gene_sets: Dict[str, List[str]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            name = parts[0].strip()
            genes = [token.strip() for token in parts[2:] if token.strip()]
            if genes:
                gene_sets[name] = genes
    return gene_sets


def _parse_weighted_table(path: Path, weight_column: str) -> Mapping[str, Dict[str, float]]:
    delimiter = "," if path.suffix.lower() == ".csv" else "\t"
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        required = {"pathway", "gene", weight_column}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Weighted pathway table {path} missing columns {sorted(missing)} "
                f"(available: {reader.fieldnames})"
            )
        collection: Dict[str, Dict[str, float]] = {}
        for row in reader:
            pathway = row["pathway"].strip()
            gene = row["gene"].strip()
            if not pathway or not gene:
                continue
            try:
                weight = float(row[weight_column])
            except (ValueError, TypeError):
                continue
            gene_map = collection.setdefault(pathway, {})
            gene_map[gene] = weight
    return collection


def _load_pathway_resource(
    resource: Optional[str],
    method: str,
    weight_column: str,
) -> Mapping[str, Mapping[str, float]]:
    """Load pathway definitions as {pathway: {gene: weight}}."""
    if resource is None:
        return {}

    resource_path = Path(resource).expanduser()
    if not resource_path.exists():
        logger.warning("Pathway resource {} not found; using fallback gene chunks.", resource_path)
        return {}

    method = method.lower()
    if resource_path.suffix.lower() in {".csv", ".tsv"}:
        weighted = _parse_weighted_table(resource_path, weight_column)
        return weighted

    raw_sets = _parse_gmt(resource_path)
    if method == "progeny":
        # PROGENy 需要权重；若没有则默认权重 1
        return {name: {gene: 1.0 for gene in genes} for name, genes in raw_sets.items()}
    return {name: {gene: 1.0 for gene in genes} for name, genes in raw_sets.items()}


def _build_pathway_definitions(
    adata: "sc.AnnData",
    resource: Optional[str],
    method: str,
    *,
    min_genes: int,
    max_genes: int,
    weight_column: str,
) -> List[PathwayDefinition]:
    """Create pathway definitions mapped onto AnnData genes."""
    genes = list(adata.var_names)
    gene_to_index = {gene.upper(): idx for idx, gene in enumerate(genes)}
    n_genes = len(genes)

    raw_collection = _load_pathway_resource(resource, method, weight_column)
    if not raw_collection:
        # fallback：按顺序分块
        if method not in _FALLBACK_PATHWAY_NOTICE:
            logger.info(
                "Using fallback pathway chunks because no external resource provided for method '{}'.",
                method,
            )
            _FALLBACK_PATHWAY_NOTICE.add(method)
        chunk_size = max(min_genes, max(1, n_genes // 10))
        raw_collection = {
            f"chunk_{idx + 1}": {
                gene: 1.0
                for gene in genes[idx * chunk_size : (idx + 1) * chunk_size]
            }
            for idx in range(max(1, n_genes // chunk_size))
        }

    pathway_defs: List[PathwayDefinition] = []
    for name, gene_weights in raw_collection.items():
        indices: List[int] = []
        weights: List[float] = []
        for gene, weight in gene_weights.items():
            key = gene.upper()
            if key not in gene_to_index:
                continue
            indices.append(gene_to_index[key])
            weights.append(float(weight))

        if len(indices) < min_genes or len(indices) > max_genes:
            continue
        indices_arr = np.array(sorted(set(indices)), dtype=int)
        if indices_arr.size == 0:
            continue
        weights_arr = np.array(weights, dtype=float)
        # Re-align weights to sorted indices
        if weights_arr.size != indices_arr.size:
            aligned = []
            index_to_weight = {idx: weight for idx, weight in zip(indices, weights)}
            for idx in indices_arr.tolist():
                aligned.append(index_to_weight.get(idx, 0.0))
            weights_arr = np.array(aligned, dtype=float)
        complement = np.setdiff1d(np.arange(n_genes), indices_arr, assume_unique=True)
        pathway_defs.append(
            PathwayDefinition(
                name=name,
                indices=indices_arr,
                complement=complement,
                weights=weights_arr,
            )
        )

    if not pathway_defs:
        logger.warning(
            "No pathways passed filtering for method '%s' (min=%d, max=%d).",
            method,
            min_genes,
            max_genes,
        )
    return pathway_defs


def _compute_ssgsea_scores(
    dense_matrix: np.ndarray,
    pathways: Sequence[PathwayDefinition],
    *,
    alpha: float,
) -> Tuple[np.ndarray, List[str]]:
    if not pathways:
        return np.zeros((dense_matrix.shape[0], 0), dtype=float), []

    # Higher expression -> higher rank score
    ranks = _rank_transform(dense_matrix)
    ranks = 1.0 - ranks  # convert to descending order (high expression -> large value)
    ranks = np.power(np.clip(ranks, 0.0, 1.0), alpha)
    total_mass = ranks.sum(axis=1, keepdims=True)

    scores: List[np.ndarray] = []
    names: List[str] = []
    for pathway in pathways:
        idx = pathway.indices
        comp = pathway.complement
        if idx.size == 0 or comp.size == 0:
            continue
        hit_mass = ranks[:, idx].sum(axis=1) / idx.size
        miss_mass = np.where(
            comp.size > 0,
            (total_mass.squeeze(-1) - ranks[:, idx].sum(axis=1)) / comp.size,
            0.0,
        )
        scores.append((hit_mass - miss_mass).reshape(-1, 1))
        names.append(f"ssgsea_{pathway.name}")

    if not scores:
        return np.zeros((dense_matrix.shape[0], 0), dtype=float), []
    return np.hstack(scores), names


def _compute_gsva_scores(
    dense_matrix: np.ndarray,
    pathways: Sequence[PathwayDefinition],
    *,
    alpha: float,
) -> Tuple[np.ndarray, List[str]]:
    if not pathways:
        return np.zeros((dense_matrix.shape[0], 0), dtype=float), []

    zscores = _zscore(dense_matrix)
    scores: List[np.ndarray] = []
    names: List[str] = []

    for pathway in pathways:
        idx = pathway.indices
        comp = pathway.complement
        if idx.size == 0:
            continue
        inside = zscores[:, idx].mean(axis=1)
        outside = (
            zscores[:, comp].mean(axis=1) if comp.size > 0 else np.zeros(zscores.shape[0])
        )
        diff = (inside - outside) * alpha
        scores.append(diff.reshape(-1, 1))
        names.append(f"gsva_{pathway.name}")

    if not scores:
        return np.zeros((dense_matrix.shape[0], 0), dtype=float), []
    return np.hstack(scores), names


def _compute_progeny_scores(
    dense_matrix: np.ndarray,
    pathways: Sequence[PathwayDefinition],
) -> Tuple[np.ndarray, List[str]]:
    if not pathways:
        return np.zeros((dense_matrix.shape[0], 0), dtype=float), []

    scores: List[np.ndarray] = []
    names: List[str] = []

    for pathway in pathways:
        idx = pathway.indices
        if idx.size == 0:
            continue
        weights = pathway.weights
        if weights is None or weights.size == 0:
            weights = np.ones(idx.size, dtype=float)
        norm = np.sum(np.abs(weights)) or 1.0
        projection = dense_matrix[:, idx] * (weights / norm)
        scores.append(np.sum(projection, axis=1, keepdims=True))
        names.append(f"progeny_{pathway.name}")

    if not scores:
        return np.zeros((dense_matrix.shape[0], 0), dtype=float), []
    return np.hstack(scores), names


def _compute_pathway_activity(
    adata: "sc.AnnData",
    *,
    method: str,
    resource: Optional[str],
    min_genes: int,
    max_genes: int,
    alpha: float,
    weight_column: str,
) -> Tuple[np.ndarray, List[str], Dict[str, float]]:
    dense = _to_dense(adata.X)
    pathways = _build_pathway_definitions(
        adata,
        resource,
        method,
        min_genes=min_genes,
        max_genes=max_genes,
        weight_column=weight_column,
    )
    if not pathways:
        return np.zeros((dense.shape[0], 0), dtype=float), [], {
            "pathway_method": method,
            "n_pathway_features": 0,
            "pathway_resource": resource,
        }

    method = method.lower()
    if method == "ssgsea":
        matrix, names = _compute_ssgsea_scores(dense, pathways, alpha=alpha)
    elif method == "gsva":
        matrix, names = _compute_gsva_scores(dense, pathways, alpha=alpha)
    else:
        matrix, names = _compute_progeny_scores(dense, pathways)

    stats = {
        "pathway_method": method,
        "n_pathway_features": int(matrix.shape[1]),
        "pathway_feature_mean": float(np.mean(matrix)),
        "pathway_feature_std": float(np.std(matrix)),
        "pathway_resource": resource,
    }
    return matrix, names, stats


def _resolve_resource(resource_map: Optional[Mapping[str, str]], method: str, default: Optional[str]) -> Optional[str]:
    if not resource_map:
        return default
    keys = [
        method,
        method.lower(),
        method.upper(),
        method.capitalize(),
        "default",
    ]
    for key in keys:
        if key in resource_map:
            return resource_map[key]
    return default


def compute_gene_plus_pathway_features_multi(
    adata: "sc.AnnData",
    *,
    methods: Sequence[str],
    resource_map: Optional[Mapping[str, str]] = None,
    default_resource: Optional[str] = None,
    max_components: int = 50,
    min_genes: int = 5,
    max_genes: int = 500,
    alpha: float = 0.75,
    weight_column: str = "weight",
) -> Tuple[np.ndarray, list[str], Dict[str, float]]:
    """Augment gene features with scores from multiple pathway methods."""
    if not methods:
        return compute_gene_only_features(adata, max_components=max_components)

    method_sequence = []
    for item in methods:
        if not item:
            continue
        name = str(item).strip()
        if not name:
            continue
        lname = name.lower()
        if lname in method_sequence:
            continue
        method_sequence.append(lname)

    if not method_sequence:
        return compute_gene_only_features(adata, max_components=max_components)

    gene_matrix, gene_names, diagnostics = compute_gene_only_features(
        adata,
        max_components=max_components,
    )

    feature_matrix = gene_matrix
    feature_names = list(gene_names)
    total_pathway_dims = 0
    pathway_stats: Dict[str, float] = {}

    for method in method_sequence:
        resource = _resolve_resource(resource_map, method, default_resource)
        matrix, names, stats = _compute_pathway_activity(
            adata,
            method=method,
            resource=resource,
            min_genes=min_genes,
            max_genes=max_genes,
            alpha=alpha,
            weight_column=weight_column,
        )
        total_pathway_dims += int(stats.get("n_pathway_features", 0))
        pathway_stats[f"pathway_{method}_n_features"] = int(stats.get("n_pathway_features", 0))
        pathway_stats[f"pathway_{method}_resource"] = resource
        if matrix.shape[1] > 0:
            feature_matrix = np.hstack([feature_matrix, matrix])
            feature_names.extend(names)

    diagnostics = {
        **diagnostics,
        **pathway_stats,
        "mode": "gene_plus_pathway",
        "pathway_methods": method_sequence,
        "n_pathway_features_total": total_pathway_dims,
        "n_features": int(feature_matrix.shape[1]),
        "n_samples": int(feature_matrix.shape[0]),
    }
    return feature_matrix, feature_names, diagnostics


def compute_gene_plus_pathway_features(
    adata: "sc.AnnData",
    *,
    method: str = "ssgsea",
    resource: Optional[str] = None,
    max_components: int = 50,
    min_genes: int = 5,
    max_genes: int = 500,
    alpha: float = 0.75,
    weight_column: str = "weight",
) -> Tuple[np.ndarray, list[str], Dict[str, float]]:
    """Legacy helper to combine genes with a single pathway method."""
    return compute_gene_plus_pathway_features_multi(
        adata,
        methods=[method],
        resource_map={"default": resource} if resource else None,
        default_resource=resource,
        max_components=max_components,
        min_genes=min_genes,
        max_genes=max_genes,
        alpha=alpha,
        weight_column=weight_column,
    )
