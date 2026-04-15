# Import statements
import argparse
import numpy as np
import pandas as pd
import oddt
import oddt.pandas as opd
from oddt.fingerprints import PLEC
from joblib import Parallel, delayed
from tqdm import tqdm

PLEC_SIZE = 4092

# Module-level receptor — set in main() before parallel call so workers
# inherit it via fork without pickling (same pattern as original script)
receptor = None

def make_plec(mol):
    return PLEC(mol, protein=receptor, size=PLEC_SIZE,
                depth_protein=5, depth_ligand=1,
                distance_cutoff=4.5, sparse=False)

def parse_args():
    parser = argparse.ArgumentParser(description="Generate PLEC fingerprint features from docked molecules.")
    parser.add_argument("csv_file", help="Path to the bioactivity data CSV file (must have a 'mol_name' column)")
    parser.add_argument("ligands_mol2", help="Path to the docked ligands .mol2 file")
    parser.add_argument("receptor_mol2", help="Path to the receptor .mol2 file")
    parser.add_argument("output_csv", help="Path for the output PLEC features CSV file")
    parser.add_argument("--num_cores", type=int, default=-1, help="Number of cores for parallel processing (default: all available)")
    return parser.parse_args()

def main():
    global receptor
    args = parse_args()

    # Load bioactivity data
    bioactivity_data = pd.read_csv(args.csv_file, dtype={'mol_name': 'object'})
    if 'mol_name' not in bioactivity_data.columns:
        raise ValueError(f"CSV file must contain a 'mol_name' column. Found: {list(bioactivity_data.columns)}")
    n_dupes = bioactivity_data.duplicated(subset=['mol_name']).sum()
    if n_dupes:
        print(f"Warning: {n_dupes} duplicate mol_name entries in CSV; keeping first occurrence.")
    bioactivity_data = bioactivity_data.drop_duplicates(subset=['mol_name'])

    # Load and merge docked ligands with bioactivity data
    mol2_file = opd.read_mol2(args.ligands_mol2)
    mol2_file.columns = ['mol', 'mol_name']
    mol2_data = mol2_file.merge(bioactivity_data, how='left', on='mol_name')

    # Load receptor into module-level global so workers inherit it via fork
    receptor = next(oddt.toolkit.readfile('mol2', args.receptor_mol2))

    # Generate PLEC features in parallel
    features = Parallel(n_jobs=args.num_cores, backend="multiprocessing")(
        delayed(make_plec)(mol) for mol in tqdm(mol2_data['mol'])
    )

    # Save PLEC features to CSV
    column_names = [f"PLEC_{i}" for i in range(PLEC_SIZE)]
    PLEC_df = pd.DataFrame(features, columns=column_names)
    PLEC_df.index = mol2_data['mol_name'].values
    PLEC_df.index.name = 'Molecule_Name'
    PLEC_df.to_csv(args.output_csv)

if __name__ == "__main__":
    main()
