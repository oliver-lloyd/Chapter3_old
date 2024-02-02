import pandas as pd
import pubchempy as pcp
from os import listdir

entities = pd.read_csv('../../../entity_ids.del', header=None, sep='\t')[1].values
drugs = [ent for ent in entities if ent.startswith("CID")]


drugs_int = [int(drug.split('CID')[1]) for drug in drugs]  # Pubchem takes integer IDs
canonical_smiles = []
for drug, cid in zip(drugs, drugs_int):
    print(len(canonical_smiles))
    smile = pcp.Compound.from_cid(cid).canonical_smiles
    canonical_smiles.append([drug, smile])

out = pd.DataFrame(canonical_smiles, columns=['drug', 'canonical smiles'])
out.to_csv('canonical_smiles.csv', index=False)