# Import statements
import numpy as np
import pandas as pd
import oddt
import oddt.pandas as opd
from oddt.fingerprints import PLEC
from joblib import Parallel, delayed
from tqdm import tqdm

# Load bioactivity data
PfDHODH_data = pd.read_csv("../data/PfDHODH-data.csv", dtype={'mol_name': 'object'})

#Provide the pathway to docked molecules mol2 file
mol2_file = opd.read_mol2("../data/PfDHODH-docked.mol2")
mol2_file.columns = ['mol', 'mol_name']
mol2_data = mol2_file.merge(PfDHODH_data.drop_duplicates(subset = ['mol_name']), how = 'left', on = 'mol_name')

#Extract the structures of all molecules
mols = mol2_data['mol']

# Provide the pathway to the training and test set target/receptor structure
receptor = next(oddt.toolkit.readfile('mol2', '../data/PfDHODH-protein.mol2'))

# Define a function to generate PLEC features 
def parallel_plec(mol):
    feature = PLEC(mol, protein = receptor, size = 4092, 
                  depth_protein = 5, depth_ligand = 1,
                  distance_cutoff = 4.5, sparse = False)
    return feature

# Generate PLEC features using 20 cores
num_cores = 20
features = Parallel(n_jobs = num_cores, backend = "multiprocessing")(delayed(parallel_plec)(mol) for mol in tqdm(mols))

# Save PLEC features to a .csv file
# Create column names
column_names = [f"PLEC_{i}" for i in range(4092)]

# Convert the list of arrays to a DataFrame
PLEC_df = pd.DataFrame(features, columns=column_names)

# Add molecule names as the index
PLEC_df.index = PfDHODH_data['mol_name']
PLEC_df.index.name = 'Molecule_Name'
PLEC_df.to_csv("../features/PLEC_features.csv")



