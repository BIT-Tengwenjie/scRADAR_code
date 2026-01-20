"""Utilities for loading and using drug fingerprint caches."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from loguru import logger

try:
    import anndata as ad  # type: ignore
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    ad = None
    pd = None

from .pathways import _compute_pathway_activity


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


def _normalize_vector(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    finite_mask = np.isfinite(arr)
    if not finite_mask.any():
        return np.zeros_like(arr)
    mean = float(np.mean(arr[finite_mask]))
    std = float(np.std(arr[finite_mask]))
    if std <= 0:
        std = 1.0
    return (arr - mean) / std


def _normalized_mechanism_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


@dataclass
class FingerprintVector:
    values: np.ndarray
    feature_names: List[str]
    pathway_profile: Optional[Dict[str, float]] = None
    static_tokens: Optional[List[Tuple[str, float]]] = None


class DrugFingerprintCache:
    """Loads static/dynamic drug fingerprints from a cached npz file."""

    def __init__(
        self,
        cache_path: Path,
        *,
        methods: Sequence[str],
        resource_map: Optional[Mapping[str, str]],
        default_resource: Optional[str],
        min_genes: int = 5,
        max_genes: int = 500,
        alpha: float = 0.75,
        weight_column: str = "weight",
    ) -> None:
        cache_path = cache_path.expanduser().resolve()
        if not cache_path.exists():
            raise FileNotFoundError(f"Drug fingerprint cache not found: {cache_path}")

        payload = np.load(cache_path, allow_pickle=True)
        self.drug_keys: List[str] = [str(key) for key in payload["drug_keys"]]
        self.drug_names: List[str] = [
            str(name) for name in payload.get("drug_names", payload["drug_keys"])
        ]
        self.key_to_index: Dict[str, int] = {key: idx for idx, key in enumerate(self.drug_keys)}

        self.static_features: Optional[np.ndarray] = payload.get("static_features")
        self.static_feature_names: List[str] = [
            str(name) for name in payload.get("static_feature_names", [])
        ]

        self.dynamic_signatures: Optional[np.ndarray] = payload.get("dynamic_signatures")
        self.dynamic_gene_names: List[str] = [
            str(name) for name in payload.get("dynamic_gene_names", [])
        ]
        self.dynamic_feature_names: List[str] = []

        self.methods = [str(m).lower() for m in methods if str(m).strip()]
        self.resource_map = resource_map or {}
        self.default_resource = default_resource
        self.min_genes = int(min_genes)
        self.max_genes = int(max_genes)
        self.alpha = float(alpha)
        self.weight_column = weight_column

        self._pathway_matrix: Optional[np.ndarray] = None
        self._pathway_profiles: Dict[str, Dict[str, float]] = {}

    def _resolve_resource(self, method: str) -> Optional[str]:
        if not self.resource_map:
            return self.default_resource
        keys = [
            method,
            method.lower(),
            method.upper(),
            method.capitalize(),
            "default",
        ]
        for key in keys:
            if key in self.resource_map:
                return self.resource_map[key]
        return self.default_resource

    def _ensure_dynamic_pathways(self) -> Optional[np.ndarray]:
        if self._pathway_matrix is not None:
            return self._pathway_matrix
        if (
            self.dynamic_signatures is None
            or self.dynamic_signatures.size == 0
            or not self.dynamic_gene_names
        ):
            return None
        if ad is None or pd is None:
            logger.warning(
                "anndata/pandas not available; cannot compute dynamic fingerprint pathways."
            )
            return None
        if not self.methods:
            return None

        obs = pd.DataFrame(index=self.drug_keys)
        var = pd.DataFrame(index=self.dynamic_gene_names)
        adata = ad.AnnData(self.dynamic_signatures.astype(np.float32), obs=obs, var=var)

        matrices: List[np.ndarray] = []
        names: List[str] = []
        for method in self.methods:
            resource = self._resolve_resource(method)
            matrix, method_names, _ = _compute_pathway_activity(
                adata,
                method=method,
                resource=resource,
                min_genes=self.min_genes,
                max_genes=self.max_genes,
                alpha=self.alpha,
                weight_column=self.weight_column,
            )
            if matrix.shape[1] == 0:
                continue
            prefix = f"drugfp_{method}_"
            normalized_names = []
            for item in method_names:
                token = item.split("_", 1)[-1]
                normalized_names.append(prefix + token)
            names.extend(normalized_names)
            matrices.append(matrix)

        if not matrices:
            return None

        self._pathway_matrix = np.hstack(matrices)
        self.dynamic_feature_names = names
        for row_idx, key in enumerate(self.drug_keys):
            profile: Dict[str, float] = {}
            for col_idx, feat_name in enumerate(self.dynamic_feature_names):
                norm_name = _normalized_mechanism_name(feat_name.split("_", 2)[-1])
                profile[norm_name] = float(self._pathway_matrix[row_idx, col_idx])
            self._pathway_profiles[key] = profile
        return self._pathway_matrix

    def get(
        self,
        drug_name: str,
        *,
        static_weight: float,
        dynamic_weight: float,
    ) -> Optional[FingerprintVector]:
        key = _slugify(drug_name)
        idx = self.key_to_index.get(key)
        if idx is None:
            return None

        segments: List[np.ndarray] = []
        feature_names: List[str] = []
        static_tokens: Optional[List[Tuple[str, float]]] = None

        if (
            static_weight > 0
            and self.static_features is not None
            and self.static_features.size > 0
        ):
            vec = _normalize_vector(self.static_features[idx])
            segments.append(static_weight * vec)
            feature_names.extend([f"drugfp_static_{name}" for name in self.static_feature_names])
            token_pairs: List[Tuple[str, float]] = []
            indices = np.argsort(vec)[::-1][: min(10, vec.size)]
            for i in indices:
                token_pairs.append((self.static_feature_names[i], float(vec[i])))
            static_tokens = token_pairs

        dynamic_matrix = self._ensure_dynamic_pathways()
        pathway_profile: Optional[Dict[str, float]] = None
        if (
            dynamic_weight > 0
            and dynamic_matrix is not None
            and dynamic_matrix.size > 0
            and self.dynamic_feature_names
        ):
            dvec = _normalize_vector(dynamic_matrix[idx])
            segments.append(dynamic_weight * dvec)
            feature_names.extend(self.dynamic_feature_names)
            pathway_profile = self._pathway_profiles.get(key)

        if not segments:
            return None

        values = np.concatenate(segments, axis=0)
        return FingerprintVector(
            values=values,
            feature_names=feature_names,
            pathway_profile=pathway_profile,
            static_tokens=static_tokens,
        )


def load_cache_for_flags(flags: Mapping[str, object]) -> Optional[DrugFingerprintCache]:
    cache_path = flags.get("drug_fingerprint_cache")
    if not cache_path:
        return None
    methods = flags.get("drug_fingerprint_methods") or ["progeny"]
    resource_map = flags.get("drug_fingerprint_resource_map")
    if isinstance(resource_map, str):
        resource_map = {"default": resource_map}
    default_resource = flags.get("pathway_resource")
    try:
        return DrugFingerprintCache(
            Path(cache_path),
            methods=methods,
            resource_map=resource_map,
            default_resource=default_resource,
            min_genes=int(flags.get("pathway_min_genes", 5)),
            max_genes=int(flags.get("pathway_max_genes", 500)),
            alpha=float(flags.get("pathway_rank_alpha", 0.75)),
            weight_column=str(flags.get("progeny_weight_column", "weight")),
        )
    except FileNotFoundError:
        logger.warning("Drug fingerprint cache %s not found; skipping.", cache_path)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to load fingerprint cache: %s", exc)
    return None


def normalize_drug_name(name: str) -> str:
    """Public helper for scripts to reuse the same normalization logic."""
    return _slugify(name)
