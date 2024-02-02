import pandas as pd
from numpy import array_split
from scipy.spatial.distance import cosine, euclidean
from os import listdir
from itertools import combinations

# Load embeddings
embeds_file = 'drug_embeds.csv'
if embeds_file in listdir():
    embeds = pd.read_csv(embeds_file, index_col=0)
else:
    embeds_df = pd.read_csv('../../neighbour_vector_aggregation/vector_estimates.csv')
    embeds_df = embeds_df.query('vector_method == "actual embedding"')
    embeds = embeds[[str(i) for i in range(256)]]
    embeds.index = embeds_df.drug
    embeds.to_csv(embeds_file)

# Calculate vector similarities
results = []
pairs = combinations(embeds.index, 2)
for drug1, drug2 in pairs:
    embed1 = embeds.loc[drug1]
    embed2 = embeds.loc[drug2]
    cos = 1 - cosine(embed1, embed2)
    euc = euclidean(embed1, embed2)
    embed1_head, embed1_tail = array_split(embed1, 2)
    embed2_head, embed2_tail = array_split(embed2, 2)
    cos_head = 1 - cosine(embed1_head, embed2_head)
    cos_tail = 1 - cosine(embed1_tail, embed2_tail)
    results.append([drug1, drug2, cos, euc, cos_head, cos_tail])

out = pd.DataFrame(results, columns=['drug1', 'drug2', 'cosine_sim', 'euc_dist', 'cosine_head', 'cosine_tail'])

# Save
out.to_csv('embedding_similarity.csv', index=False)
