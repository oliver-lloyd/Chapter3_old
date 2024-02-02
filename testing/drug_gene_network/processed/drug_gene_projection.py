import pandas as pd
from itertools import combinations

# Read bipartite graph
drug_target = pd.read_csv('../raw/bio-decagon-targets.csv')
drug_drug = {}

# Perform bipartite projection using Drugs' shared target genes
for gene, subdf in drug_target.groupby('Gene'):
    n_edges_in = len(subdf)
    if n_edges_in > 1:
        drugs = subdf.STITCH.values
        for pair in combinations(drugs, 2):
            pair_key = frozenset(pair)
            norm_weight = 1/n_edges_in
            if pair_key in drug_drug:
                drug_drug[pair_key]['normal_weight'] += norm_weight
                drug_drug[pair_key]['weight'] += 1
            else:
                drug_drug[pair_key] = {
                    'normal_weight': norm_weight,
                    'weight': 1
                }

flattened = [list(key) + [drug_drug[key]['weight'], drug_drug[key]['normal_weight']] for key in drug_drug]
out = pd.DataFrame(flattened, columns=['drug1', 'drug2', 'weight', 'normalised_weight'])
out.to_csv('drug_drug_edges.tsv', sep='\t', index=False)
