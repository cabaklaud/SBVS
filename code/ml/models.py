from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost.sklearn import XGBRegressor

SUPPORTED_MODELS = ("RF", "XGB", "SVM")


def get_search_space(model_name, trial):
    """
    Define the Optuna hyperparameter search space for each model.

    Parameters
    ----------
    model_name : str
        One of 'RF', 'XGB', 'SVR'.
    trial : optuna.trial.Trial

    Returns
    -------
    dict of hyperparameters
    """
    if model_name == "RF":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 2000),
            "criterion": trial.suggest_categorical(
                "criterion", ["squared_error", "friedman_mse", "poisson"]
            ),
        }
    elif model_name == "XGB":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_categorical("max_depth", [5, 6, 7, 8, 9, 10]),
        }
    elif model_name == "SVM":
        return {
            "C": trial.suggest_int("C", 1, 10),
            "kernel": trial.suggest_categorical("kernel", ["rbf", "poly", "sigmoid"]),
        }
    else:
        raise ValueError(f"model_name must be one of {SUPPORTED_MODELS}, got '{model_name}'")


def build_model(model_name, params, random_state=42, n_jobs=40):
    """
    Instantiate a model with the given hyperparameters.

    Parameters
    ----------
    model_name : str
        One of 'RF', 'XGB', 'SVM'.
    params : dict
        Hyperparameters (e.g. from Optuna best_params).
    random_state : int
        Random seed (ignored for SVM).
    n_jobs : int
        Parallelism (ignored for SVM).

    Returns
    -------
    sklearn-compatible estimator
    """
    if model_name == "RF":
        return RandomForestRegressor(
            n_estimators=params["n_estimators"],
            criterion=params["criterion"],
            max_features="sqrt",
            n_jobs=n_jobs,
            random_state=random_state,
        )
    elif model_name == "XGB":
        return XGBRegressor(
            objective="reg:squarederror",
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            n_jobs=n_jobs,
            random_state=random_state,
        )
    elif model_name == "SVM":
        return SVR(C=params["C"], kernel=params["kernel"])
    else:
        raise ValueError(f"model_name must be one of {SUPPORTED_MODELS}, got '{model_name}'")
