import numpy as np
import optuna
import optunahub
from sklearn.model_selection import StratifiedKFold

from ml.metrics import nef_score
from ml.models import build_model, get_search_space

optuna.logging.set_verbosity(optuna.logging.WARNING)


def run_study(model_name, X_train, y_train, n_trials=10, n_splits=3, n_jobs=40):
    """
    Run an Optuna hyperparameter search using the HEBO sampler.

    Cross-validation uses StratifiedKFold on binarised potency labels
    (active = potency >= 6) and optimises NEF@10%.

    Parameters
    ----------
    model_name : str
        One of 'RF', 'XGB', 'SVR'.
    X_train : np.ndarray
    y_train : np.ndarray
        Continuous potency values.
    n_trials : int
    n_splits : int
        Number of CV folds.
    n_jobs : int
        Passed to build_model for parallelism.

    Returns
    -------
    optuna.Study
    """
    y_bin = (y_train >= 6).astype(int)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    def objective(trial):
        params = get_search_space(model_name, trial)
        model = build_model(model_name, params, random_state=42, n_jobs=n_jobs)

        nef_scores = []
        for train_idx, val_idx in cv.split(X_train, y_bin):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]
            model.fit(X_tr, y_tr)
            nef_scores.append(nef_score(y_val, model.predict(X_val), top_percent=10))

        return float(np.mean(nef_scores))

    module = optunahub.load_module("samplers/hebo")
    sampler = module.HEBOSampler()
    study = optuna.create_study(sampler=sampler, direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    print(f"Best params: {study.best_params}")
    print(f"Best CV NEF@10%: {study.best_value:.4f}")

    return study
