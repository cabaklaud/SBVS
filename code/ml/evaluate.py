import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from ml.models import build_model

N_SEEDS = 10


def train_and_evaluate(
    model_name,
    best_params,
    features,
    labels,
    heldout_cluster,
    feature_type,
    output_dir,
    pu_decoys=True,
    n_jobs=40,
):
    """
    Train a model with 10 different random seeds and save ranked predictions
    for both test sets.

    Parameters
    ----------
    model_name : str
        One of 'RF', 'XGB', 'SVR'.
    best_params : dict
        Hyperparameters from Optuna.
    features : dict
        Keys: 'train', 'full_test', 'act_decoys' — each an np.ndarray.
    labels : dict
        Keys: 'train_potency', 'full_test_potency', 'act_decoys_potency',
               'full_test_class', 'act_decoys_class'.
    heldout_cluster : int
    feature_type : str
        e.g. 'PLEC', 'Morgan', 'PLEC+Morgan' — used to build the output path.
    output_dir : str
        Root directory for screening results.
    pu_decoys : bool
    n_jobs : int
    """
    decoy_tag = "pu_decoys" if pu_decoys else "no_pu_decoys"
    base = os.path.join(output_dir, decoy_tag, f"heldout_cluster_{heldout_cluster}", feature_type)
    full_test_dir  = os.path.join(base, "full_test_set")
    act_decoys_dir = os.path.join(base, "smaller_test_set")
    os.makedirs(full_test_dir,  exist_ok=True)
    os.makedirs(act_decoys_dir, exist_ok=True)

    model_tag = f"r{model_name}"

    for seed in tqdm(range(N_SEEDS), desc=f"{model_name} seeds"):
        model = build_model(model_name, best_params, random_state=seed, n_jobs=n_jobs)
        model.fit(np.array(features["train"]), labels["train_potency"])

        for X_test, potency, classes, out_dir in [
            (features["full_test"],  labels["full_test_potency"],  labels["full_test_class"],  full_test_dir),
            (features["act_decoys"], labels["act_decoys_potency"], labels["act_decoys_class"], act_decoys_dir),
        ]:
            preds = model.predict(np.array(X_test))
            result = pd.DataFrame({
                "Predicted_Potency": preds,
                "Real_Potency":      potency,
                "Real_Class":        classes,
            })
            result = result.sort_values(by="Predicted_Potency", ascending=False)
            result.to_csv(os.path.join(out_dir, f"PfDHODH-{model_tag}-seed{seed + 1}.csv"))
