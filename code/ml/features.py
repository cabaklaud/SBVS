import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from ml.data import split_features

SUPPORTED_FEATURES = ("PLEC", "weighted_ECIF", "multishelled_ECIF", "Morgan", "PLEC+Morgan")


def load_plec(plec_csv):
    """Load PLEC feature matrix from CSV, indexed by Molecule_Name."""
    return pd.read_csv(plec_csv, dtype={"Molecule_Name": "object"}, index_col="Molecule_Name")


def load_ecif(ecif_csv):
    """Load ECIF feature matrix from CSV, indexed by Molecule_Name."""
    return pd.read_csv(ecif_csv, dtype={"Molecule_Name": "object"}, index_col="Molecule_Name")


def generate_morgan(smiles_list, radius=2, fp_size=512):
    """Generate Morgan fingerprint vectors for a list of SMILES strings."""
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius,
        countSimulation=False,
        includeChirality=True,
        fpSize=fp_size,
    )
    fingerprints = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        fingerprints.append(np.array(generator.GetFingerprint(mol)))
    return fingerprints


def get_features(feature_type, train_df, full_test_df, test_act_decoys_df, plec_csv=None, ecif_csv=None):
    """
    Build train and test feature arrays for the requested feature type.

    Parameters
    ----------
    feature_type : str
        One of SUPPORTED_FEATURES.
    train_df, full_test_df, test_act_decoys_df : pd.DataFrame
        Split dataframes with mol_name and SMILES columns.
    plec_csv : str or None
        Path to PLEC features CSV. Required for 'PLEC' and 'PLEC+Morgan'.
    ecif_csv : str or None
        Path to ECIF features CSV. Required for 'weighted_ECIF' and 'multishelled_ECIF'.

    Returns
    -------
    dict with keys: train, full_test, act_decoys — each an np.ndarray
    """
    if feature_type not in SUPPORTED_FEATURES:
        raise ValueError(f"feature_type must be one of {SUPPORTED_FEATURES}, got '{feature_type}'")

    if feature_type in ("PLEC", "PLEC+Morgan"):
        if plec_csv is None:
            raise ValueError("plec_csv must be provided for PLEC-based features")
        plec_df = load_plec(plec_csv)
        train_plec,      _ = split_features(plec_df, train_df, full_test_df)
        _, full_test_plec  = split_features(plec_df, train_df, full_test_df)
        _, act_decoys_plec = split_features(plec_df, train_df, test_act_decoys_df)

    if feature_type in ("Morgan", "PLEC+Morgan"):
        train_morgan      = generate_morgan(train_df["SMILES"].tolist())
        full_test_morgan  = generate_morgan(full_test_df["SMILES"].tolist())
        act_decoys_morgan = generate_morgan(test_act_decoys_df["SMILES"].tolist())

    if feature_type in ("weighted_ECIF", "multishelled_ECIF"):
        if ecif_csv is None:
            raise ValueError("ecif_csv must be provided for ECIF-based features")
        ecif_df = load_ecif(ecif_csv)
        train_ecif,      _ = split_features(ecif_df, train_df, full_test_df)
        _, full_test_ecif  = split_features(ecif_df, train_df, full_test_df)
        _, act_decoys_ecif = split_features(ecif_df, train_df, test_act_decoys_df)

    if feature_type == "PLEC":
        return {
            "train":      train_plec,
            "full_test":  full_test_plec,
            "act_decoys": act_decoys_plec,
        }
    elif feature_type == "Morgan":
        return {
            "train":      np.array(train_morgan),
            "full_test":  np.array(full_test_morgan),
            "act_decoys": np.array(act_decoys_morgan),
        }
    elif feature_type in ("weighted_ECIF", "multishelled_ECIF"):
        return {
            "train":      train_ecif,
            "full_test":  full_test_ecif,
            "act_decoys": act_decoys_ecif,
        }
    else:  # PLEC+Morgan
        return {
            "train":      np.array([np.hstack((p, m)) for p, m in zip(train_plec,      train_morgan)]),
            "full_test":  np.array([np.hstack((p, m)) for p, m in zip(full_test_plec,  full_test_morgan)]),
            "act_decoys": np.array([np.hstack((p, m)) for p, m in zip(act_decoys_plec, act_decoys_morgan)]),
        }
