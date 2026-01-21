# Project Overview
Single-cell drug response prediction and training pipeline. Current run config = `config.py` base + `overrides/run_global.json` overrides: dual pathway features (ssGSEA + PROGENy), drug mechanism fingerprints, ProtoMechanism model, and a single global F1 grid threshold.

## Current Configuration Highlights
- Model: `proto_mech`, `proto_num=13`, `proto_topk=4`, `proto_mech_lambda=1.3`, `proto_diversity_lambda=0.12`.
- Pathway features: `pathway_methods=["ssgsea","progeny"]` with resources
  - ssGSEA: `data/resource/c2.cp.reactome.v2025.1.Hs.symbols.gmt`
  - PROGENy: `data/resource/progeny_human_top500.tsv`
  Other params: `pathway_min_genes=5`, `pathway_max_genes=500`, `pathway_rank_alpha=0.75`.
- Drug fingerprints: enabled, method `["progeny"]`, static/dynamic weights 0.55/0.45, cache `data/resources/drug_fingerprint_cache.npz`.
- Mechanism FiLM: enabled (hidden dim 48) to modulate pathway channels by drug mechanism.
- Threshold: `threshold_strategy="f1_grid"`, `threshold_beta=1.1`, single global threshold.
- I/O paths: `DATA_ROOT`, `RESULTS_ROOT`, `LOG_ROOT` in `config.py`; inputs from `data/processed/*.h5ad`; outputs at `${RESULTS_ROOT}/<run-name>/`.

## Method Highlights
- Dual pathway representation: ssGSEA (Reactome GMT) + PROGENy (perturbation weight TSV), covering metabolic and signaling responses.
- Drug mechanism fingerprints: PROGENy mechanism vectors (static + dynamic) concatenated into the model.
- ProtoMechanism: top-k sparse routing, capacity/diversity regularizers, and mechanism FiLM for drug-aware prototypes.
- Decision: global F1 grid threshold; evaluation uses 95% t-based confidence intervals.

## Data Flow and Training
1. Data loading: read `data/processed/*.h5ad`, parse labels using `DATASET_METADATA`.
2. Feature construction: `features/pipeline.py` computes ssGSEA + PROGENy scores and concatenates gene expression; `features/drug_fingerprints.py` loads PROGENy fingerprints to build static/dynamic priors.
3. Model training: `models/factory.py` builds ProtoMechanism; `code/run.py` trains with sparse routing, capacity/diversity regularization, and FiLM conditioning.
4. Threshold selection: validation F1 grid search for a single global threshold.
5. Inference and save: apply global threshold, compute metrics (ACC/F1/AUROC/AUPRC), and write artifacts.

## Outputs (${RESULTS_ROOT}/<run-name>/)
- `results.csv`: per-dataset metrics.
- `summary.json`: threshold and aggregate metrics.
- `prototype_diagnostics.json`: prototype usage, capacity, routing distribution.
- `model_checkpoint.npz`: model parameters.

## Run Example
```bash
PIPELINE_OVERRIDES="$(cat overrides/run_global.json)" \
python code/run.py --config config.py --run-name proto_mech_global
```

## Directory
```
config.py                 # Paths/dataset metadata & PIPELINE_FLAGS
overrides/run_global.json # Override items for this run (model/pathway/fingerprint/threshold)
code/
  run.py                  # Training and inference entry
  data_loader.py          # Data loading and preprocessing
  features/               # Pathway and fingerprint feature builders
  models/                 # ProtoMechanism
  postprocess/            # Threshold selection
scripts/                  # Fingerprint cache tools
data/                     # processed inputs, results outputs, resource files
```
