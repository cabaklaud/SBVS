import numpy as np
import pandas as pd


def get_train_test_split(df, heldout_cluster, pu_decoys=True):
    """
    Split bioactivity data into train and two test sets based on cluster ID.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset with columns: activity, Cluster_ID, mol_name, potency, SMILES.
    heldout_cluster : int
        Cluster ID (0-3) whose actives form the test set.
    pu_decoys : bool
        If False, property-unmatched decoys (mol_name starting with 'PU_DECOY')
        are excluded from all splits.

    Returns
    -------
    train_df : pd.DataFrame
    full_test_df : pd.DataFrame
        Test actives + all decoys.
    test_act_decoys_df : pd.DataFrame
        Test actives + only the decoys paired to the heldout cluster.
    """
    if not pu_decoys:
        df = df[~df["mol_name"].str.startswith("PU_DECOY")]

    train_actives = df[(df["activity"] == "Active") & (df["Cluster_ID"] != heldout_cluster)]
    test_actives  = df[(df["activity"] == "Active") & (df["Cluster_ID"] == heldout_cluster)]

    train_inactives   = df[(df["activity"] == "Inactive") & (df["Cluster_ID"] == -1)]
    all_decoys        = df[(df["activity"] == "Inactive") & (df["Cluster_ID"] != -1)]
    test_act_decoys   = df[(df["activity"] == "Inactive") & (df["Cluster_ID"] == heldout_cluster)]

    train_df           = pd.concat([train_actives, train_inactives],  ignore_index=True)
    full_test_df       = pd.concat([test_actives,  all_decoys],       ignore_index=True)
    test_act_decoys_df = pd.concat([test_actives,  test_act_decoys],  ignore_index=True)

    return train_df, full_test_df, test_act_decoys_df


def extract_labels(train_df, full_test_df, test_act_decoys_df):
    """
    Extract potency and class label arrays for train and both test sets.

    Returns a dict with keys:
        train_potency, full_test_potency, act_decoys_potency,
        train_class,   full_test_class,   act_decoys_class
    """
    return {
        "train_potency":      train_df["potency"].values,
        "full_test_potency":  full_test_df["potency"].values,
        "act_decoys_potency": test_act_decoys_df["potency"].values,
        "train_class":        train_df["activity"].values,
        "full_test_class":    full_test_df["activity"].values,
        "act_decoys_class":   test_act_decoys_df["activity"].values,
    }


def split_features(feature_df, train_df, test_df):
    """
    Index into a feature DataFrame by mol_name to get train/test arrays.

    Parameters
    ----------
    feature_df : pd.DataFrame
        Rows indexed by mol_name.
    train_df, test_df : pd.DataFrame
        Must contain a 'mol_name' column.

    Returns
    -------
    train_features, test_features : np.ndarray
    """
    train_features = feature_df.loc[train_df["mol_name"]].values
    test_features  = feature_df.loc[test_df["mol_name"]].values
    return train_features, test_features
