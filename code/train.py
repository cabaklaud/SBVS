"""
Train and evaluate a regression model for virtual screening.

Example
-------
python train.py \
    --model RF \
    --features PLEC+Morgan \
    --heldout_cluster 3 \
    --data ../data/PfDHODH-data.csv \
    --plec_features ../data/features/PLEC_features.csv \
    --output_dir ../results
"""
import argparse

import pandas as pd

from ml.data import get_train_test_split, extract_labels
from ml.features import get_features, SUPPORTED_FEATURES
from ml.models import SUPPORTED_MODELS
from ml.tuning import run_study
from ml.evaluate import train_and_evaluate


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate a regression model for virtual screening.")
    parser.add_argument("--model",           required=True, choices=SUPPORTED_MODELS,   help="Model to train")
    parser.add_argument("--features",        required=True, choices=SUPPORTED_FEATURES, help="Feature type to use")
    parser.add_argument("--heldout_cluster", required=True, type=int, choices=[0, 1, 2, 3], help="Cluster ID to hold out as test actives")
    parser.add_argument("--data",            required=True, help="Path to bioactivity CSV (PfDHODH-data.csv)")
    parser.add_argument("--plec_features",   default=None,  help="Path to PLEC features CSV (required for PLEC or PLEC+Morgan)")
    parser.add_argument("--no_pu_decoys",    action="store_true", help="Exclude property-unmatched decoys (default: include them)")
    parser.add_argument("--output_dir",      default="./screening-results", help="Root directory for output CSVs")
    parser.add_argument("--n_trials",        type=int, default=10,  help="Number of Optuna trials (default: 10)")
    parser.add_argument("--n_jobs",          type=int, default=40,  help="CPU cores for model training (default: 40)")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading data from {args.data}")
    df = pd.read_csv(args.data)
    pu_decoys = not args.no_pu_decoys
    train_df, full_test_df, test_act_decoys_df = get_train_test_split(df, args.heldout_cluster, pu_decoys=pu_decoys)
    print(f"  Train: {len(train_df)} | Full test: {len(full_test_df)} | Smaller test: {len(test_act_decoys_df)}")

    print(f"Building {args.features} features...")
    features = get_features(
        args.features, train_df, full_test_df, test_act_decoys_df,
        plec_csv=args.plec_features,
    )
    labels = extract_labels(train_df, full_test_df, test_act_decoys_df)

    print(f"Tuning {args.model} with {args.n_trials} Optuna trials...")
    study = run_study(
        args.model,
        features["train"],
        labels["train_potency"],
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
    )

    print(f"Training final models and evaluating...")
    train_and_evaluate(
        model_name=args.model,
        best_params=study.best_params,
        features=features,
        labels=labels,
        heldout_cluster=args.heldout_cluster,
        feature_type=args.features,
        output_dir=args.output_dir,
        pu_decoys=pu_decoys,
        n_jobs=args.n_jobs,
    )

    decoy_tag = "pu_decoys" if pu_decoys else "no_pu_decoys"
    print(f"Done. Results saved to {args.output_dir}/{decoy_tag}/heldout_cluster_{args.heldout_cluster}/{args.features}/")


if __name__ == "__main__":
    main()
