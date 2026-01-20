#!/usr/bin/env python3
"""Build hybrid drug fingerprints by combining GDSC metadata and LINCS signatures."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import h5py  # type: ignore
import numpy as np
import pandas as pd
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import CONFIG, RESOURCES  # noqa: E402
from code.features.drug_fingerprints import normalize_drug_name  # noqa: E402


def _gather_dataset_drugs() -> Dict[str, str]:
    drugs: Dict[str, str] = {}
    datasets_cfg = CONFIG.get("datasets_config") or {}
    for dataset, meta in datasets_cfg.items():
        name = str(meta.get("drug_name", "")).strip()
        if not name:
            continue
        key = normalize_drug_name(name)
        drugs[key] = name
    return drugs


def _load_table(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    logger.info("Loading {}", path)
    return pd.read_csv(path, sep="\t", low_memory=False)


def build_static_fingerprints(screened_path: Path, drug_keys: Dict[str, str]) -> Tuple[np.ndarray, List[str], Dict[str, np.ndarray]]:
    df = pd.read_csv(screened_path)
    df["drug_key"] = df["DRUG_NAME"].astype(str).map(normalize_drug_name)
    df = df[df["drug_key"].isin(drug_keys.keys())].copy()
    token_sets = {
        "target": set(),
        "pathway": set(),
    }

    def _tokens(value: str) -> List[str]:
        if not isinstance(value, str):
            return []
        return [token.strip() for token in re.split(r"[;,/]", value) if token.strip()]

    for _, row in df.iterrows():
        for token in _tokens(row.get("TARGET", "")):
            token_sets["target"].add(token)
        for token in _tokens(row.get("TARGET_PATHWAY", "")):
            token_sets["pathway"].add(token)

    feature_names: List[str] = []
    for prefix, tokens in token_sets.items():
        for token in sorted(tokens):
            feature_names.append(f"{prefix}::{token}")

    static_dict: Dict[str, np.ndarray] = {}
    for _, row in df.iterrows():
        vector = np.zeros(len(feature_names), dtype=np.float32)
        token_map = {
            "target": _tokens(row.get("TARGET", "")),
            "pathway": _tokens(row.get("TARGET_PATHWAY", "")),
        }
        for idx, name in enumerate(feature_names):
            prefix, token = name.split("::", 1)
            if token in token_map.get(prefix, []):
                vector[idx] = 1.0
        static_dict[row["drug_key"]] = vector

    matrix = np.stack([static_dict.get(key, np.zeros(len(feature_names), dtype=np.float32)) for key in drug_keys.keys()])
    return matrix, feature_names, static_dict


def load_sig_metadata(lincs_dir: Path, pattern: str) -> pd.DataFrame:
    frames = []
    for path in sorted(lincs_dir.glob(pattern)):
        frames.append(pd.read_csv(path, sep="\t", low_memory=False))
    if not frames:
        raise FileNotFoundError(f"No metadata files matched {pattern}")
    return pd.concat(frames, ignore_index=True)


def _resolve_data_dataset(data_group) -> "h5py.Dataset":
    obj = data_group
    while True:
        if hasattr(obj, "keys"):
            keys = list(obj.keys())
            if not keys:
                raise RuntimeError("DATA group does not contain any dataset.")
            obj = obj[keys[0]]
            continue
        return obj


def build_dynamic_signatures(lincs_dir: Path, drug_keys: Dict[str, str]) -> Tuple[np.ndarray, List[str]]:
    sig_info = load_sig_metadata(lincs_dir, "GSE*_sig_info*.txt*")
    sig_info = sig_info[sig_info["pert_type"] == "trt_cp"].copy()
    sig_info["drug_key"] = sig_info["pert_iname"].map(normalize_drug_name)
    sig_info = sig_info[sig_info["drug_key"].isin(drug_keys.keys())]
    if sig_info.empty:
        raise RuntimeError("No overlapping signatures between LINCS and dataset drugs.")

    gctx_path = next((p for p in lincs_dir.glob("level5_beta_trt_cp_*.gctx")), None)
    if not gctx_path:
        raise FileNotFoundError("Cannot locate level5_beta_trt_cp_* gctx file.")

    logger.info("Indexing GCTX {}", gctx_path)
    with h5py.File(gctx_path, "r") as handle:
        col_ids = [cid.decode("utf-8") for cid in handle["0"]["META"]["COL"]["id"][()]]
        row_ids = [rid.decode("utf-8") for rid in handle["0"]["META"]["ROW"]["id"][()]]

    col_index = {cid: idx for idx, cid in enumerate(col_ids)}
    gene_info_path = next((p for p in lincs_dir.glob("GSE*_gene_info*.txt*")), None)
    if not gene_info_path:
        raise FileNotFoundError("Gene info file not found under LINCS directory.")
    gene_info = pd.read_csv(gene_info_path, sep="\t", low_memory=False)
    gene_id_col = None
    for candidate in ("gene_id", "pr_gene_id", "GENE_ID", "pr_gene_id"):
        if candidate in gene_info.columns:
            gene_id_col = candidate
            break
    gene_symbol_col = None
    for candidate in ("gene_symbol", "pr_gene_symbol", "GENE_SYMBOL"):
        if candidate in gene_info.columns:
            gene_symbol_col = candidate
            break
    if gene_id_col is None or gene_symbol_col is None:
        raise RuntimeError(f"Could not find gene_id/symbol columns in {gene_info_path}")
    gene_map = {str(row[gene_id_col]): row[gene_symbol_col] for _, row in gene_info.iterrows()}
    gene_names = [gene_map.get(rid, rid) for rid in row_ids]

    dynamic_vectors: Dict[str, np.ndarray] = {}
    row_count = len(row_ids)
    col_count = len(col_ids)
    with h5py.File(gctx_path, "r") as handle:
        matrix = _resolve_data_dataset(handle["0"]["DATA"])
        current_shape = matrix.shape
        if current_shape == (row_count, col_count):
            def _load_signature_columns(indices: np.ndarray) -> np.ndarray:
                return np.asarray(matrix[:, indices], dtype=np.float32)
        elif current_shape == (col_count, row_count):
            def _load_signature_columns(indices: np.ndarray) -> np.ndarray:
                # Dataset stores signatures along axis 0; transpose to genes x sigs.
                return np.asarray(matrix[indices, :], dtype=np.float32).T
        else:
            raise RuntimeError(
                f"Unexpected DATA matrix shape {current_shape}; "
                f"expected {(row_count, col_count)} or {(col_count, row_count)}."
            )
        for key in drug_keys.keys():
            subset = sig_info[sig_info["drug_key"] == key]
            sig_ids = [sig for sig in subset["sig_id"] if sig in col_index]
            if not sig_ids:
                continue
            idxs = np.array(sorted({col_index[sig] for sig in sig_ids}), dtype=int)
            signature = _load_signature_columns(idxs)
            dynamic_vectors[key] = np.nanmedian(signature, axis=1)

    dynamic_matrix = []
    for key in drug_keys.keys():
        dynamic_matrix.append(dynamic_vectors.get(key, np.zeros(len(row_ids), dtype=np.float32)))
    return np.stack(dynamic_matrix), gene_names


def main() -> None:
    parser = argparse.ArgumentParser(description="Build hybrid drug fingerprint cache.")
    parser.add_argument("--output", default=CONFIG["pipeline_flags"].get("drug_fingerprint_cache"), help="Path to output npz cache.")
    parser.add_argument("--screened-meta", default=RESOURCES.get("drug_static_meta") if "RESOURCES" in globals() else None, help="Path to screened_compounds metadata.")
    parser.add_argument("--lincs-dir", default=RESOURCES.get("lincs_dir") if "RESOURCES" in globals() else None, help="Directory containing LINCS Level5 files.")
    args = parser.parse_args()

    output_path = Path(args.output or "drug_fingerprint_cache.npz").expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    screened_path = Path(args.screened_meta or "").expanduser().resolve()
    lincs_dir = Path(args.lincs_dir or "").expanduser().resolve()
    if not screened_path.exists():
        raise FileNotFoundError(f"screened_compounds file not found: {screened_path}")
    if not lincs_dir.exists():
        raise FileNotFoundError(f"LINCS directory not found: {lincs_dir}")

    drug_keys = _gather_dataset_drugs()
    if not drug_keys:
        raise RuntimeError("No drugs detected from dataset configuration (missing drug_name?).")

    logger.info("Building static fingerprints for %d drugs", len(drug_keys))
    static_matrix, static_feature_names, static_dict = build_static_fingerprints(screened_path, drug_keys)

    logger.info("Building dynamic signatures from LINCS")
    dynamic_matrix, gene_names = build_dynamic_signatures(lincs_dir, drug_keys)

    payload = {
        "drug_keys": np.array(list(drug_keys.keys())),
        "drug_names": np.array(list(drug_keys.values())),
        "static_features": static_matrix.astype(np.float32),
        "static_feature_names": np.array(static_feature_names, dtype=object),
        "dynamic_signatures": dynamic_matrix.astype(np.float32),
        "dynamic_gene_names": np.array(gene_names, dtype=object),
    }
    np.savez_compressed(output_path, **payload)
    logger.info("Fingerprint cache written to %s", output_path)


if __name__ == "__main__":
    main()
