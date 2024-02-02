import pandas as pd
from kge.model import KgeModel
from kge.util.io import load_checkpoint
from itertools import combinations
from scipy.spatial.distance import cosine

# Load embeddings
model_path = '../../../../Chapter2/analysis/experiments/selfloops/simple/20230929-111630-simple_selfloops/00021/checkpoint_best.pt'
checkpoint = load_checkpoint(model_path)
data_path = '/Users/fu19841/Documents/thesis_analysis/kge/data/selfloops'
checkpoint['config'].set('dataset.name', data_path)
model = KgeModel.create_from(checkpoint)
node_embeds = model.state_dict()['_entity_embedder._embeddings.weight']

# Get node info
node_list = pd.read_csv(
    checkpoint['config'].get('dataset.name') + '/entity_ids.del',
    sep='\t', header=None
)
node_name_to_index = {row[1]: row[0] for _, row in node_list.iterrows()}
drugs = [node for node in node_name_to_index.keys() if node.startswith('CID')]

# Pairwise vector comparison
dyads = combinations(drugs, 2)
results = []
for drug1, drug2 in dyads:
    embed1 = node_embeds[node_name_to_index[drug1]]
    embed2 = node_embeds[node_name_to_index[drug2]]
    cos_sim = 1 - cosine(embed1, embed2)
    results.append([drug1, drug2, cos_sim])

# Save
out = pd.DataFrame(results, columns=['drug1', 'drug2', 'cosine_similarity'])
out.to_csv('embedding_cosines.csv', index=False)