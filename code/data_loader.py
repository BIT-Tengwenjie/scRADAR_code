"""Unified helpers for loading bulk and single-cell datasets.

This module centralises IO logic for scRADAR training.
It exposes a `BulkDataLoader` utility for working with GDSC-style source data
and lightweight helpers to load the single-cell AnnData targets referenced in
``config.py``.  Keeping the logic here avoids duplicating parsing code across
multiple training scripts.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, Union

import numpy as np
import pandas as pd
try:
    from loguru import logger  # type: ignore
except ImportError:  # pragma: no cover
    import logging

    logger = logging.getLogger(__name__)

from config import CONFIG

try:
    import scanpy as sc  # type: ignore
except ImportError:  # pragma: no cover
    sc = None

# Default filenames expected under the bulk ``source`` directory.
DEFAULT_BULK_FILES: Dict[str, str] = {
    "expression": "Cell_line_RMA_proc_basalExp.txt",
    "gdsc1": "GDSC1_fitted_dose_response_27Oct23.xlsx",
    "gdsc2": "GDSC2_fitted_dose_response_27Oct23.xlsx",
    "cell_metadata": "Cell_Lines_Details.xlsx",
    "drug_metadata": "screened_compounds_rel_8.5.csv",
}


GROUP_CANDIDATES = [
    "group_id",
    "patient_id",
    "patient",
    "donor",
    "sample_id",
    "sample",
    "orig.ident",
]

BATCH_CANDIDATES = [
    "batch",
    "batch_id",
    "library",
    "library_id",
]

CLUSTER_CANDIDATES = [
    "cluster",
    "clusters",
    "leiden",
    "louvain",
]


def _select_obs_series(adata: "sc.AnnData", candidates: Iterable[str]) -> Optional[pd.Series]:
    for column in candidates:
        if column in adata.obs and adata.obs[column].notna().any():
            return adata.obs[column].astype(str).fillna("NA")
    return None


def _default_label_series(adata: "sc.AnnData", prefix: str) -> pd.Series:
    return pd.Series(
        [f"{prefix}_{idx}" for idx in range(adata.n_obs)],
        index=adata.obs.index,
        dtype="string",
    )


def _ensure_metadata_columns(adata: "sc.AnnData") -> None:
    if "group_id" not in adata.obs.columns:
        group_series = _select_obs_series(adata, GROUP_CANDIDATES)
        if group_series is None:
            group_series = _default_label_series(adata, "group")
        adata.obs["group_id"] = group_series.to_numpy()

    if "batch_label" not in adata.obs.columns:
        batch_series = _select_obs_series(adata, BATCH_CANDIDATES)
        if batch_series is None:
            batch_series = _default_label_series(adata, "batch")
        adata.obs["batch_label"] = batch_series.to_numpy()

    if "cluster" not in adata.obs.columns:
        cluster_series = _select_obs_series(adata, CLUSTER_CANDIDATES)
        if cluster_series is None:
            cluster_series = _default_label_series(adata, "cluster")
        adata.obs["cluster"] = cluster_series.to_numpy()


def _ensure_standardized_response(
    adata: "sc.AnnData",
    dataset_name: str,
    dataset_config: Dict,
) -> None:
    """Ensure ``adata.obs`` contains a binary ``standardized_response`` column."""
    if "standardized_response" in adata.obs.columns:
        adata.obs["standardized_response"] = (
            adata.obs["standardized_response"].astype(float).astype(int)
        )
        return

    label_column = dataset_config.get("label_column")
    positive_values = dataset_config.get("positive_values")
    negative_values = dataset_config.get("negative_values")

    if not label_column or not positive_values:
        raise KeyError(
            f"Dataset {dataset_name} lacks 'standardized_response' and "
            "no label mapping (label_column/positive_values) is defined in CONFIG."
        )
    if label_column not in adata.obs.columns:
        raise KeyError(
            f"Dataset {dataset_name}: label column '{label_column}' not found in AnnData.obs."
        )

    series = adata.obs[label_column]
    normalised = series.astype(str).str.strip().str.lower()
    normalised = normalised.where(series.notna(), other=np.nan)

    pos = {str(value).strip().lower() for value in positive_values}
    if not pos:
        raise ValueError(
            f"Dataset {dataset_name}: 'positive_values' must contain at least one entry."
        )

    if negative_values:
        neg = {str(value).strip().lower() for value in negative_values}
    else:
        neg = {value for value in normalised.dropna().unique() if value not in pos}

    valid = pos.union(neg)
    invalid_mask = normalised.notna() & ~normalised.isin(valid)
    if invalid_mask.any():
        invalid_values = sorted(set(normalised[invalid_mask]))
        raise ValueError(
            f"Dataset {dataset_name}: unexpected label values {invalid_values} "
            f"in column '{label_column}'. Define 'positive_values'/'negative_values' accordingly."
        )

    adata.obs["standardized_response"] = normalised.isin(pos).astype(int)
    logger.info(
        "Dataset {}: derived standardized_response from column '{}' (positive={}).",
        dataset_name,
        label_column,
        sorted(pos),
    )


class ResultsManager:
    """Utility for writing experiment logs and metric tables."""

    def __init__(
        self,
        run_name: str,
        config: Dict = CONFIG,
        *,
        fieldnames: Optional[Iterable[str]] = None,
    ) -> None:
        self.base_dir = (Path(config["results_root"]).expanduser() / run_name).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.base_dir / "log.txt"
        self.csv_path = self.base_dir / "results.csv"
        self._fieldnames: Optional[list[str]] = list(fieldnames) if fieldnames else None

    def append_log(self, message: str, *, timestamp: bool = True) -> None:
        stamp = (
            f"[{datetime.now().isoformat(timespec='seconds')}] "
            if timestamp
            else ""
        )
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{stamp}{message}\n")

    def append_row(self, row: Dict[str, object]) -> None:
        if self._fieldnames is None:
            if self.csv_path.exists() and self.csv_path.stat().st_size > 0:
                with self.csv_path.open("r", encoding="utf-8", newline="") as handle:
                    reader = csv.reader(handle)
                    try:
                        self._fieldnames = next(reader)
                    except StopIteration:
                        self._fieldnames = list(row.keys())
            else:
                self._fieldnames = list(row.keys())

        missing = set(row.keys()) - set(self._fieldnames)
        if missing:
            raise ValueError(
                f"Row contains unexpected columns {missing}; expected {self._fieldnames}."
            )

        is_new_file = not self.csv_path.exists() or self.csv_path.stat().st_size == 0
        with self.csv_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self._fieldnames)
            if is_new_file:
                writer.writeheader()
            writer.writerow({key: row.get(key, "") for key in self._fieldnames})


def get_results_manager(
    run_name: str,
    config: Dict = CONFIG,
    *,
    fieldnames: Optional[Iterable[str]] = None,
) -> ResultsManager:
    """Factory helper mirroring ``ResultsManager`` construction."""
    return ResultsManager(run_name, config=config, fieldnames=fieldnames)


def _resolve_bulk_source_dir(config: Dict = CONFIG) -> Path:
    """Locate the bulk source directory using the project configuration."""
    data_root = Path(config["data_root"])
    source_dir = config.get("bulk_source_dir")
    if source_dir:
        return Path(source_dir).expanduser()
    return (data_root / "source").expanduser()


def list_available_single_cell_datasets(config: Dict = CONFIG) -> Iterable[str]:
    """Return the dataset names discovered in the project configuration."""
    return tuple(config.get("dataset_names", []))


def load_single_cell_dataset(
    dataset_name: str,
    config: Dict = CONFIG,
    *,
    copy: bool = True,
) -> sc.AnnData:
    """Load a single-cell AnnData object by name using ``config.py``."""
    if sc is None:
        raise ImportError(
            "scanpy is required to load single-cell datasets. "
            "Install scanpy or restrict usage to bulk helpers."
        )
    dataset_config = config.get("datasets_config", {}).get(dataset_name)
    if not dataset_config:
        raise KeyError(f"Dataset {dataset_name} is not defined in CONFIG.")

    adata_path = Path(dataset_config["adata_path"])
    if not adata_path.exists():
        raise FileNotFoundError(f"AnnData file not found: {adata_path}")

    logger.debug(f"Loading AnnData target dataset {dataset_name} from {adata_path}.")
    adata = sc.read_h5ad(adata_path)
    _ensure_standardized_response(adata, dataset_name, dataset_config)
    _ensure_metadata_columns(adata)
    return adata.copy() if copy else adata


@dataclass
class BulkDataLoader:
    """Helper for accessing bulk (cell-line) pharmacogenomic data."""

    source_dir: Union[str, Path]
    expression_filename: str = DEFAULT_BULK_FILES["expression"]
    gdsc1_filename: str = DEFAULT_BULK_FILES["gdsc1"]
    gdsc2_filename: str = DEFAULT_BULK_FILES["gdsc2"]
    cell_metadata_filename: str = DEFAULT_BULK_FILES["cell_metadata"]
    drug_metadata_filename: str = DEFAULT_BULK_FILES["drug_metadata"]

    def __post_init__(self) -> None:
        self.source_dir = Path(self.source_dir).expanduser()
        self._expression: Optional[pd.DataFrame] = None
        self._response_table: Optional[pd.DataFrame] = None
        self._cell_metadata: Optional[pd.DataFrame] = None
        self._drug_metadata: Optional[pd.DataFrame] = None

        required_files = self._iter_required_files()
        missing = [name for name, path in required_files.items() if not path.exists()]
        if missing:
            detail = ", ".join(f"{name} -> {required_files[name]}" for name in missing)
            raise FileNotFoundError(
                f"Missing expected bulk source files: {detail}"
            )

    @classmethod
    def from_config(cls, config: Dict = CONFIG, **overrides) -> "BulkDataLoader":
        """Create a loader using the paths referenced in the project config."""
        source_dir = overrides.pop("source_dir", _resolve_bulk_source_dir(config))
        file_overrides = DEFAULT_BULK_FILES.copy()
        file_overrides.update(config.get("bulk_source_files", {}))
        file_overrides.update({k: overrides.pop(k) for k in list(overrides.keys()) if k in file_overrides})

        return cls(
            source_dir=source_dir,
            expression_filename=file_overrides["expression"],
            gdsc1_filename=file_overrides["gdsc1"],
            gdsc2_filename=file_overrides["gdsc2"],
            cell_metadata_filename=file_overrides["cell_metadata"],
            drug_metadata_filename=file_overrides["drug_metadata"],
        )

    def _iter_required_files(self) -> Dict[str, Path]:
        return {
            "expression": self.source_dir / self.expression_filename,
            "gdsc1": self.source_dir / self.gdsc1_filename,
            "gdsc2": self.source_dir / self.gdsc2_filename,
            "cell_metadata": self.source_dir / self.cell_metadata_filename,
            "drug_metadata": self.source_dir / self.drug_metadata_filename,
        }

    @staticmethod
    def _normalise_cosmic_column(column: str) -> str:
        if not column.startswith("DATA."):
            return column
        remainder = column[5:]
        try:
            # handle columns like DATA.905954.1 by converting to int first
            cosmic = str(int(float(remainder)))
        except ValueError:
            cosmic = remainder
        return cosmic

    def load_expression(self, *, copy: bool = True) -> pd.DataFrame:
        """Load the RMA-normalised expression matrix (genes x COSMIC IDs)."""
        if self._expression is None:
            path = self.source_dir / self.expression_filename
            logger.info(f"Loading bulk expression matrix from {path}.")
            df = pd.read_csv(path, sep="\t")
            if "GENE_SYMBOLS" not in df.columns:
                raise ValueError("Expression file missing 'GENE_SYMBOLS' column.")

            df = df.rename(columns={"GENE_SYMBOLS": "gene_symbol"})
            if "GENE_title" in df.columns:
                df = df.drop(columns=["GENE_title"])
            df = df.set_index("gene_symbol")
            df.columns = [self._normalise_cosmic_column(col) for col in df.columns]

            df = df.apply(pd.to_numeric, errors="coerce")
            df = df.T.groupby(level=0).mean().T  # average replicate columns
            df = df.groupby(level=0).mean()  # collapse duplicate gene symbols

            self._expression = df

        return self._expression.copy() if copy else self._expression

    def load_cell_metadata(self, *, copy: bool = True) -> pd.DataFrame:
        """Return metadata about each cell line."""
        if self._cell_metadata is None:
            path = self.source_dir / self.cell_metadata_filename
            logger.info(f"Loading cell-line metadata from {path}.")
            df = pd.read_excel(path)
            df = df.rename(
                columns={
                    "Sample Name": "sample_name",
                    "COSMIC identifier": "cosmic_id",
                }
            )
            df["cosmic_id"] = (
                df["cosmic_id"]
                .dropna()
                .astype(float)
                .astype(int)
                .astype(str)
            )
            self._cell_metadata = df

        return self._cell_metadata.copy() if copy else self._cell_metadata

    def load_drug_metadata(self, *, copy: bool = True) -> pd.DataFrame:
        """Return metadata about the screened compounds."""
        if self._drug_metadata is None:
            path = self.source_dir / self.drug_metadata_filename
            logger.info(f"Loading drug metadata from {path}.")
            df = pd.read_csv(path)
            df = df.rename(
                columns={
                    "DRUG_ID": "drug_id",
                    "DRUG_NAME": "drug_name",
                    "TARGET": "target",
                    "TARGET_PATHWAY": "target_pathway",
                }
            )
            df["drug_id"] = df["drug_id"].astype(int)
            self._drug_metadata = df

        return self._drug_metadata.copy() if copy else self._drug_metadata

    def load_response_table(self, *, copy: bool = True) -> pd.DataFrame:
        """Combine the GDSC1 and GDSC2 fitted IC50/AUC tables."""
        if self._response_table is None:
            tables = []
            usecols = [
                "DATASET",
                "COSMIC_ID",
                "CELL_LINE_NAME",
                "DRUG_ID",
                "DRUG_NAME",
                "LN_IC50",
                "AUC",
                "Z_SCORE",
            ]
            for fname in (self.gdsc1_filename, self.gdsc2_filename):
                path = self.source_dir / fname
                logger.info(f"Loading drug response table from {path}.")
                df = pd.read_excel(path, usecols=usecols)
                tables.append(df)

            response = pd.concat(tables, ignore_index=True)
            response = response.dropna(subset=["COSMIC_ID", "DRUG_ID", "DRUG_NAME"])

            response["COSMIC_ID"] = response["COSMIC_ID"].apply(
                lambda value: str(int(float(value))) if pd.notnull(value) else np.nan
            )
            response = response.dropna(subset=["COSMIC_ID"])
            response["DRUG_ID"] = response["DRUG_ID"].astype(int)
            response["LN_IC50"] = pd.to_numeric(response["LN_IC50"], errors="coerce")
            response["AUC"] = pd.to_numeric(response["AUC"], errors="coerce")

            response = response.rename(
                columns={
                    "COSMIC_ID": "cosmic_id",
                    "DRUG_ID": "drug_id",
                    "DRUG_NAME": "drug_name",
                    "CELL_LINE_NAME": "cell_line_name",
                    "LN_IC50": "ln_ic50",
                    "AUC": "auc",
                }
            )
            response = response.drop_duplicates(subset=["drug_id", "cosmic_id"])

            self._response_table = response

        return self._response_table.copy() if copy else self._response_table

    @staticmethod
    def _assign_labels(values: pd.Series, strategy: str, **kwargs) -> pd.Series:
        """Convert continuous LN(IC50) values into binary labels."""
        clean = values.astype(float)
        if strategy == "mean":
            threshold = kwargs.get("threshold", float(clean.mean()))
            return (clean <= threshold).astype(int)
        if strategy == "median":
            threshold = kwargs.get("threshold", float(clean.median()))
            return (clean <= threshold).astype(int)
        if strategy == "threshold":
            if "threshold" not in kwargs:
                raise ValueError("Explicit 'threshold' required for strategy='threshold'.")
            return (clean <= float(kwargs["threshold"])).astype(int)

        raise ValueError(f"Unsupported label strategy: {strategy}")

    def get_response_for_drug(
        self,
        drug: Union[int, str],
        *,
        label_strategy: str = "mean",
        **label_kwargs,
    ) -> pd.DataFrame:
        """Return response rows for a single drug with binary labels attached."""
        table = self.load_response_table(copy=False)
        if isinstance(drug, int):
            subset = table[table["drug_id"] == drug].copy()
        else:
            drug_lower = str(drug).lower()
            subset = table[table["drug_name"].str.lower() == drug_lower].copy()

        if subset.empty:
            raise KeyError(f"No response data found for drug {drug}.")

        subset = subset.dropna(subset=["ln_ic50"])

        duplicate_mask = subset["cosmic_id"].duplicated(keep=False)
        if duplicate_mask.any():
            duplicate_ids = sorted(subset.loc[duplicate_mask, "cosmic_id"].unique())
            preview = ", ".join(duplicate_ids[:10]) + (" ..." if len(duplicate_ids) > 10 else "")
            logger.warning(
                f"Aggregating {int(duplicate_mask.sum())} duplicate response entries for drug {drug} "
                f"across {len(duplicate_ids)} COSMIC IDs: {preview}"
            )

            aggregation: Dict[str, str] = {}
            mean_columns = {"ln_ic50", "auc", "z_score"}
            for column in subset.columns:
                if column == "cosmic_id":
                    continue
                if column in mean_columns or np.issubdtype(subset[column].dtype, np.number) and column not in {
                    "drug_id"
                }:
                    aggregation[column] = "mean"
                else:
                    aggregation[column] = "first"

            subset = subset.groupby("cosmic_id", as_index=False).agg(aggregation)

        subset["label"] = self._assign_labels(subset["ln_ic50"], label_strategy, **label_kwargs)
        return subset

    def build_training_set(
        self,
        drug: Union[int, str],
        *,
        label_strategy: str = "mean",
        genes: Optional[Iterable[str]] = None,
        **label_kwargs,
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """Return (X, y, metadata) for the requested drug.

        X is a DataFrame of shape (n_cell_lines, n_genes) with COSMIC IDs as the
        index.  y is a binary Series aligned to X.  Metadata contains additional
        columns (drug name, ln_ic50, etc.) for the selected cell lines.
        """
        expression = self.load_expression(copy=False)
        if genes is not None:
            missing_genes = set(genes) - set(expression.index)
            if missing_genes:
                logger.warning(
                    "Requested {} genes that are absent from the expression matrix.",
                    len(missing_genes),
                )
            expression = expression.loc[expression.index.intersection(genes)]

        response = self.get_response_for_drug(
            drug, label_strategy=label_strategy, **label_kwargs
        )
        cosmic_ids = response["cosmic_id"].astype(str).tolist()

        available_ids = [cid for cid in cosmic_ids if cid in expression.columns]
        missing_ids = sorted(set(cosmic_ids) - set(available_ids))

        if missing_ids:
            logger.warning(
                "Excluded {} cell lines without expression data: {}",
                len(missing_ids),
                ", ".join(missing_ids[:10]) + (" ..." if len(missing_ids) > 10 else ""),
            )

        if not available_ids:
            raise ValueError("No overlapping cell lines between expression and response data.")

        # Expression matrix is genes x COSMIC -> transpose for (samples x genes)
        X = expression[available_ids].T.copy()
        X.index = X.index.astype(str)

        metadata = response.set_index("cosmic_id").reindex(X.index)
        missing_label_mask = metadata["label"].isna()
        if missing_label_mask.any():
            dropped = metadata.index[missing_label_mask].tolist()
            logger.warning(
                "Dropped {} cell lines lacking response labels after alignment: {}",
                len(dropped),
                ", ".join(dropped[:10]) + (" ..." if len(dropped) > 10 else ""),
            )
            metadata = metadata.loc[~missing_label_mask]
            X = X.loc[metadata.index]

        y = metadata["label"].astype(int)

        if len(X) != len(y):
            raise ValueError(
                f"Expression/label size mismatch after alignment (X={len(X)}, y={len(y)})."
            )

        return X, y, metadata


def get_bulk_loader(config: Dict = CONFIG) -> BulkDataLoader:
    """Convenience wrapper mirroring ``BulkDataLoader.from_config``."""
    return BulkDataLoader.from_config(config=config)
