# SBVS-PfDHODH

Structure-based virtual screening pipeline for *Plasmodium falciparum* DHODH (PfDHODH), a validated antimalarial drug target. Docked protein-ligand complexes are converted into ML-ready features, then regression models are trained to rank compounds by predicted potency and evaluated for virtual screening performance (enrichment of true actives near the top of the ranked list).

## Pipeline

1. **Feature generation** (`code/generate_PLEC_features.py`) — computes PLEC (Protein-Ligand Extended Connectivity) fingerprints from a receptor `.mol2` file and a set of docked ligand `.mol2` poses.
2. **Training and evaluation** (`code/train.py`) — trains a regression model on one of several feature types, tunes hyperparameters with Optuna, and scores held-out test compounds.

Supporting modules live in `code/ml/`:

| Module | Purpose |
|---|---|
| `data.py` | Cluster-based train/test splitting and label extraction |
| `features.py` | Builds PLEC, Morgan, ECIF, or combined feature matrices |
| `models.py` | Model definitions and Optuna search spaces (RF, XGB, SVM) |
| `tuning.py` | Optuna hyperparameter search (HEBO sampler) optimising NEF@10% |
| `evaluate.py` | Multi-seed training and ranked-prediction output |
| `metrics.py` | Normalised Enrichment Factor (NEF) scoring |

## Usage

### 1. Generate PLEC features

```bash
python code/generate_PLEC_features.py \
    bioactivity_data.csv \
    docked_ligands.mol2 \
    receptor.mol2 \
    PLEC_features.csv
```

### 2. Train and evaluate a model

```bash
python code/train.py \
    --model RF \
    --features PLEC+Morgan \
    --heldout_cluster 3 \
    --data bioactivity_data.csv \
    --plec_features PLEC_features.csv \
    --output_dir results
```

- `--model`: `RF`, `XGB`, or `SVM`
- `--features`: `PLEC`, `weighted_ECIF`, `multishelled_ECIF`, `Morgan`, or `PLEC+Morgan`
- `--heldout_cluster`: cluster ID (0–3) of actives held out for testing
- `--ecif_features`: required instead of `--plec_features` for ECIF-based feature types

Each run trains the chosen model across 10 random seeds and writes ranked prediction CSVs (predicted vs. real potency and activity class) under `<output_dir>/<pu_decoys|no_pu_decoys>/heldout_cluster_<N>/<features>/`.

## Data

This repository contains only the pipeline code. Bioactivity data, docked structures, and precomputed features are not included.

## Dependencies

`pandas`, `numpy`, `scikit-learn`, `xgboost`, `optuna`, `optunahub`, `rdkit`, `oddt`, `joblib`, `tqdm`
