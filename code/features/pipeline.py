"""Feature pipeline orchestrator based on project configuration."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from loguru import logger

from config import CONFIG
from .pathways import (
    compute_gene_only_features,
    compute_gene_plus_pathway_features,
    compute_gene_plus_pathway_features_multi,
)


def build_feature_matrix(
    adata: "sc.AnnData",
    config: Dict = CONFIG,
) -> Tuple[np.ndarray, list[str], Dict[str, float]]:
    """Construct the feature matrix according to ``PIPELINE_FLAGS``."""
    flags = (config or CONFIG).get("pipeline_flags", {})
    enable_pathways = flags.get("enable_pathway_features", True)
    method = flags.get("pathway_method", "ssgsea")
    methods = flags.get("pathway_methods")
    extra_methods = flags.get("extra_pathway_methods")
    method_sequence: list[str] = []
    if methods:
        if isinstance(methods, (list, tuple, set)):
            method_sequence.extend([str(m).strip() for m in methods if str(m).strip()])
        else:
            method_sequence.append(str(methods).strip())
    else:
        method_sequence.append(str(method).strip())
    if extra_methods:
        extra_list = extra_methods if isinstance(extra_methods, (list, tuple, set)) else [extra_methods]
        for item in extra_list:
            name = str(item).strip()
            if name:
                method_sequence.append(name)
    # deduplicate preserving order
    normalized_methods = []
    for item in method_sequence:
        lname = item.lower()
        if lname and lname not in normalized_methods:
            normalized_methods.append(lname)

    if enable_pathways and normalized_methods:
        resource_map = flags.get("pathway_resource_map")
        default_resource = flags.get("pathway_resource")
        features, names, diagnostics = compute_gene_plus_pathway_features_multi(
            adata,
            methods=normalized_methods,
            resource_map=resource_map,
            default_resource=default_resource,
            min_genes=int(flags.get("pathway_min_genes", 5)),
            max_genes=int(flags.get("pathway_max_genes", 500)),
            alpha=float(flags.get("pathway_rank_alpha", 0.75)),
            weight_column=str(flags.get("progeny_weight_column", "weight")),
        )
        logger.debug("Built gene+pathway features using methods %s.", normalized_methods)
    else:
        features, names, diagnostics = compute_gene_only_features(adata)
        logger.debug("Built gene-only features (pathway features disabled).")

    diagnostics = {
        **diagnostics,
        "n_features": int(features.shape[1]),
        "n_samples": int(features.shape[0]),
    }
    return features, names, diagnostics
