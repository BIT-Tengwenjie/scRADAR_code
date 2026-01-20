"""Project configuration for scRADAR training."""

import os
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("DATA_ROOT", PROJECT_ROOT / "data")).expanduser().resolve()
RESULTS_ROOT = Path(os.environ.get("RESULTS_ROOT", DATA_ROOT / "results")).expanduser().resolve()
LOG_ROOT = Path(os.environ.get("LOG_ROOT", DATA_ROOT / "logs")).expanduser().resolve()
BULK_SOURCE_DIR = DATA_ROOT / "source"
RESOURCES = {
    "reactome_hs": str(DATA_ROOT / "resource" / "c2.cp.reactome.v2025.1.Hs.symbols.gmt"),
    "progeny_hs": str(DATA_ROOT / "resource" / "progeny_human_top500.tsv"),
}

PIPELINE_FLAGS = {
    "qc_min_genes": 200,
    "qc_min_counts": 200,
    "qc_max_mito_pct": 20.0,
    "qc_min_cells": 3,
    "enable_pathway_features": True,
    "pathway_method": "ssgsea",
    "pathway_resource": RESOURCES["reactome_hs"],
    "pathway_min_genes": 5,
    "pathway_max_genes": 500,
    "pathway_rank_alpha": 0.75,
    "progeny_weight_column": "weight",
    "enable_drug_fingerprints": True,
    "drug_fingerprint_cache": str(DATA_ROOT / "resources" / "drug_fingerprint_cache.npz"),
    "drug_fingerprint_methods": ["progeny"],
    "drug_fingerprint_static_weight": 0.55,
    "drug_fingerprint_dynamic_weight": 0.45,
    "model_type": "proto_mech",
    "proto_num": 13,
    "proto_topk": 4,
    "proto_mech_lambda": 1.3,
    "proto_diversity_lambda": 0.12,
    "proto_balance_lambda": 0.015,
    "proto_entropy_lambda": 0.004,
    "enable_mechanism_film": True,
    "mechanism_film_hidden_dim": 48,
    "cluster_train_subset_ratio": 0.95,
    "cluster_label_noise": 0.02,
    "threshold_strategy": "f1_grid",
    "threshold_beta": 1.1,
    "threshold_min_recall": 0.0,
}

TRAIN_DATASETS = [
    "GSE111014_seurat_afterAnno.h5ad",
    "GSE117872_seurat_afterAnno.h5ad",
    "GSE149214_seurat_afterAnno.h5ad",
    "GSE149383_seurat_afterAnno.h5ad",
    "GSE152469_seurat_afterAnno.h5ad",
    "GSE140440_seurat_afterAnno.h5ad",
    "GSE131984_JQ1_seurat_afterAnno.h5ad",
    "GSE131984_Pac_seurat_afterAnno.h5ad",
    "GSE131984_Pal_seurat_afterAnno.h5ad",
]
DATASET_NAMES = list(TRAIN_DATASETS)
DATASET_FILE_MAP = {name: str(Path("processed") / name) for name in DATASET_NAMES}
LABEL_OVERRIDES = {
    "GSE111014_seurat_afterAnno.h5ad": {
        "label_column": "condition",
        "positive_values": ["sensitive"],
        "negative_values": ["resistant"],
    },
    "GSE117872_seurat_afterAnno.h5ad": {
        "label_column": "condition",
        "positive_values": ["sensitive"],
        "negative_values": ["resistant"],
    },
    "GSE149214_seurat_afterAnno.h5ad": {
        "label_column": "condition",
        "positive_values": ["sensitive"],
        "negative_values": ["resistant"],
    },
    "GSE149383_seurat_afterAnno.h5ad": {
        "label_column": "condition",
        "positive_values": ["sensitive"],
        "negative_values": ["resistant"],
    },
    "GSE152469_seurat_afterAnno.h5ad": {
        "label_column": "condition",
        "positive_values": ["sensitive"],
        "negative_values": ["resistant"],
    },
    "GSE140440_seurat_afterAnno.h5ad": {
        "label_column": "condition",
        "positive_values": ["sensitive"],
        "negative_values": ["resistant"],
    },
    "GSE131984_JQ1_seurat_afterAnno.h5ad": {
        "label_column": "condition",
        "positive_values": ["sensitive"],
        "negative_values": ["resistant"],
    },
    "GSE131984_Pac_seurat_afterAnno.h5ad": {
        "label_column": "condition",
        "positive_values": ["sensitive"],
        "negative_values": ["resistant"],
    },
    "GSE131984_Pal_seurat_afterAnno.h5ad": {
        "label_column": "condition",
        "positive_values": ["sensitive"],
        "negative_values": ["resistant"],
    },
}

DATASET_METADATA = {
    "GSE111014_seurat_afterAnno.h5ad": {"drug_name": "ibrutinib", "geneset": "all", "label_strategy": "mean"},
    "GSE117872_seurat_afterAnno.h5ad": {"drug_name": "Cisplatin", "geneset": "all", "label_strategy": "mean"},
    "GSE149214_seurat_afterAnno.h5ad": {"drug_name": "Erlotinib", "geneset": "all", "label_strategy": "mean"},
    "GSE149383_seurat_afterAnno.h5ad": {"drug_name": "Erlotinib", "geneset": "all", "label_strategy": "mean"},
    "GSE152469_seurat_afterAnno.h5ad": {"drug_name": "Ibrutinib", "geneset": "all", "label_strategy": "mean"},
    "GSE140440_seurat_afterAnno.h5ad": {"drug_name": "docetaxel", "geneset": "all", "label_strategy": "mean"},
    "GSE131984_JQ1_seurat_afterAnno.h5ad": {"drug_name": "JQ1", "geneset": "all", "label_strategy": "mean"},
    "GSE131984_Pac_seurat_afterAnno.h5ad": {"drug_name": "paclitaxel", "geneset": "all", "label_strategy": "mean"},
    "GSE131984_Pal_seurat_afterAnno.h5ad": {"drug_name": "palbociclib", "geneset": "all", "label_strategy": "mean"},
}

CONFIG = {
    "data_root": DATA_ROOT,
    "target_root": DATA_ROOT,
    "bulk_source_dir": BULK_SOURCE_DIR,
    "results_root": RESULTS_ROOT,
    "log_root": LOG_ROOT,
    "dataset_names": TRAIN_DATASETS,
    "datasets_config": {
        name: {
            **{"adata_path": os.path.join(DATA_ROOT, DATASET_FILE_MAP[name])},
            **LABEL_OVERRIDES.get(name, {}),
            **DATASET_METADATA.get(name, {}),
        }
        for name in DATASET_NAMES
    },
    "seed": 47,
    "pipeline_flags": PIPELINE_FLAGS,
    "dataset_flag_overrides": {},
}
import json, os
PIPELINE_FLAGS.update(json.loads(os.environ.get("PIPELINE_OVERRIDES","{}")))
ONLY_DATASET = os.environ.get("ONLY_DATASET")
if ONLY_DATASET:
    CONFIG["dataset_names"] = [ONLY_DATASET]
