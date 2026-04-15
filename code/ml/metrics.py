import numpy as np


def nef_score(y_true, y_pred, top_percent=10, active_threshold=6):
    """
    Normalised Enrichment Factor at top_percent% of the ranked list.

    Parameters
    ----------
    y_true : array-like
        Continuous potency values.
    y_pred : array-like
        Predicted potency values used for ranking.
    top_percent : float
        Percentage of the dataset to consider as the top fraction.
    active_threshold : float
        Potency cutoff above which a compound is considered active.

    Returns
    -------
    float
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    y_bin = (y_true >= active_threshold).astype(int)
    n_mols = len(y_bin)
    n_actives = np.sum(y_bin)
    top_n = max(1, int(np.ceil(n_mols * top_percent / 100)))

    top_idx = np.argsort(-y_pred)[:top_n]
    n_actives_top = np.sum(y_bin[top_idx])
    ef_max = min(top_n, n_actives)

    return n_actives_top / ef_max if ef_max > 0 else 0.0
