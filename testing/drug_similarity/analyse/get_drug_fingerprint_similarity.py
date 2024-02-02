import pandas as pd
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols
from itertools import combinations
from rdkit.Chem import rdRascalMCES

canon_smiles = pd.read_csv('canonical_smiles.csv')
molecules = {}
for i, row in canon_smiles.iterrows():
    mol = Chem.MolFromSmiles(row['canonical smiles'])
    if not mol is None:
        molecules[row.drug] = mol
    else:
        raise ValueError(f'Failed to create molecule for drug: {row.drug}')

fprints = {x: FingerprintMols.FingerprintMol(molecules[x]) for x in molecules}

dyads = combinations(molecules.keys(), 2)
results = []
for drug1, drug2 in dyads:
    sim = DataStructs.FingerprintSimilarity(fprints[drug1], fprints[drug2])
    results.append([drug1, drug2, sim])

out = pd.DataFrame(results, columns=['drug1', 'drug2', 'fingerprint_similarity'])
out.to_csv('drug_fingerprint_sims.csv', index=False)
