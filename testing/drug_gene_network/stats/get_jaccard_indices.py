import pandas as pd
import networkx as nx
from itertools import combinations


def weighted_jaccard_index(vec_i, vec_j):
    import numpy as np
    # http://static.googleusercontent.com/media/research.google.com/en//pubs/archive/36928.pdf
    # Equation 1 from above
    mat = np.concatenate([vec_i, vec_j], axis=0)
    sum_of_maxs = np.sum(np.amax(mat, axis=0))
    if sum_of_maxs == 0:
        return 0
    else:
        sum_of_mins = np.sum(np.amin(mat, axis=0))
        return sum_of_mins / sum_of_maxs


# Load weighted drug network data
edges = pd.read_csv('../processed/drug_drug_edges.tsv', sep='\t')
g = nx.from_pandas_edgelist(
    edges,
    source='drug1', target='drug2',
    edge_attr='weight'
)
adj = nx.adjacency_matrix(g).todense()
nodes = list(g.nodes)

g_norm = nx.from_pandas_edgelist(
    edges,
    source='drug1', target='drug2',
    edge_attr='normalised_weight'
)
adj_norm = nx.adjacency_matrix(g_norm, weight='normalised_weight').todense()
nodes_norm = list(g_norm.nodes)
assert nodes_norm == nodes

# Get pairwise jaccard coefficient
num_nodes = len(nodes)
dyads = combinations(range(num_nodes), 2)
jaccard_list = []
for index1, index2 in dyads:
    vec1 = adj[index1]
    vec2 = adj[index2]
    weighted_jac = weighted_jaccard_index(vec1, vec2)
    vec1_norm = adj_norm[index1]
    vec2_norm = adj_norm[index2]
    normed_jac = weighted_jaccard_index(vec1_norm, vec2_norm)
    node1 = nodes[index1]
    node2 = nodes[index2]
    unweighted_jac = list(nx.jaccard_coefficient(g, [(node1, node2)]))
    jaccard_list.append(
        [node1, node2, weighted_jac, normed_jac, unweighted_jac[0][2]]
    )

# Save
out = pd.DataFrame(
    jaccard_list,
    columns=[
        'drug1', 'drug2',
        'weighted_jaccard', 'normalised_jaccard', 'binary_jaccard'
    ]
)
out.to_csv('jaccard_indices.csv', index=False)
