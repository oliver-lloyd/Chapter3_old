"""
Script to generate vectors based on the neighbourhood of embeddings that a
drug is located in.

Note: this script is very inefficient and should be re-written when the coder
is less tired than i am right now, particularly if drug count or method count
increases by a lot.
"""

import pandas as pd
import numpy as np
import torch
import os
from kge.model import KgeModel
from kge.util.io import load_checkpoint
from sklearn.decomposition import PCA


def get_bipartite_neighbours(edgelist, drug):
    targets = edgelist.loc[edgelist['STITCH'] == drug].Gene.values
    neighbours = edgelist.query('Gene in @targets').STITCH.unique()
    neighbours = np.delete(neighbours, np.where(neighbours == drug))  # Remove self from neighbours
    return set(neighbours)


def mean_column_value(mat):
    out_vec = []
    for i in range(mat.shape[1]):
        mean_val = np.mean([vec[i] for vec in mat])
        out_vec.append(mean_val)
    return out_vec


# Load embeddings
model_path = '../../../Chapter2/analysis/experiments/selfloops/simple/20230929-111630-simple_selfloops/00021/checkpoint_best.pt'
checkpoint = load_checkpoint(model_path)
data_path = '/Users/fu19841/Documents/thesis_analysis/kge/data/selfloops'
checkpoint['config'].set('dataset.name', data_path)
model = KgeModel.create_from(checkpoint)

# Get node info
node_embeds = model.state_dict()['_entity_embedder._embeddings.weight']
node_list = pd.read_csv(
    checkpoint['config'].get('dataset.name') + '/entity_ids.del',
    sep='\t', header=None
)
node_name_to_index = {row[1]: row[0] for _, row in node_list.iterrows()}
node_index_to_name = {row[0]: row[1] for _, row in node_list.iterrows()}

# Get relation info
rel_embeds = model.state_dict()['_relation_embedder._embeddings.weight']
rel_list = pd.read_csv(
    checkpoint['config'].get('dataset.name') + '/relation_ids.del',
    sep='\t', header=None
)
rel_name_to_index = {row[1]: row[0] for _, row in rel_list.iterrows()}
rel_index_to_name = {row[0]: row[1] for _, row in rel_list.iterrows()}

# Get list of drugs
drugs = [key for key in node_name_to_index.keys() if key.startswith('CID')]

# Load drug target edges
drug_target = pd.read_csv('../../../Chapter2/data/raw/bio-decagon-targets.csv')

outfile_name = 'vector_estimates.csv'
if outfile_name not in os.listdir():
    results = pd.DataFrame(
        columns=['drug', 'vector_method', 'num_neighbours'] + [str(i) for i in range(256)]
    )
    results.to_csv(outfile_name)
else:
    results = pd.read_csv(outfile_name)

for test_drug in drugs:

    # Get info of bipartite neighbour drugs (targeting the same genes)
    neighbour_nodes = get_bipartite_neighbours(drug_target, test_drug)
    n_neighbours = len(neighbour_nodes)

    # Skip if no neighbours to learn from, otherwise get relevant matrix
    if n_neighbours == 0:
        continue
    else:
        neighbours_index = [node_name_to_index[node] for node in neighbour_nodes]
        neighbours_embeds = node_embeds[neighbours_index]

    # TODO: selecting 'stranger' stuff below should only be run if either..
    # ..'generalised inverse' or 'least squares' has not been run for this drug

    # Randomly select equal number of non-neighbour drugs
    all_strangers = set(drugs) - neighbour_nodes
    stranger_nodes = np.random.choice(list(all_strangers), n_neighbours, replace=False)
    strangers_index = [node_name_to_index[node] for node in stranger_nodes]
    strangers_embeds = node_embeds[strangers_index]

    # Combine all selected embeddings
    test_mat = torch.cat((neighbours_embeds, strangers_embeds))

    # Create adjacency vector to approximate with factorisation
    adj_vec = np.array([1 for _ in neighbour_nodes] + [0 for _ in stranger_nodes])

    ######################################
    # Solving Ax=b for x using generalised inverse of rectangular embeddings
    # Matrix test_mat is rectangular so have to take the generalised inverse to solve for my_vec..
    # .. where test_mat x my_vec = adj_vec
    method = 'Generalised inverse'
    already_run = len(results.query(f'drug == "{test_drug}" & vector_method == "{method}"'))
    if not already_run:
        rect_inverse = np.linalg.pinv(test_mat)
        my_vec = np.dot(rect_inverse, adj_vec)
        result = [test_drug, method, n_neighbours] + my_vec.tolist()
        results.loc[len(results)] = result
        results.to_csv(outfile_name, index=False)

    #######################################
    # Solving Ax=b for x (as above) but using least squares instead
    method = 'Least squares'
    already_run = len(results.query(f'drug == "{test_drug}" & vector_method == "{method}"'))
    if not already_run:
        my_vec = np.linalg.lstsq(test_mat, adj_vec)[0]
        result = [test_drug, method, n_neighbours] + my_vec.tolist()
        results.loc[len(results)] = result
        results.to_csv(outfile_name, index=False)

    #######################################
    # Take principal eigenvec of neighbour embeddings
    method = '1st eigenvec'
    already_run = len(results.query(f'drug == "{test_drug}" & vector_method == "{method}"'))
    if not already_run:
        if n_neighbours >= 2:
            my_vec = PCA().fit(neighbours_embeds).components_[1]
            result = [test_drug, method, n_neighbours] + my_vec.tolist()
        else:
            result = [test_drug, method, n_neighbours] + [np.nan for _ in range(len(my_vec))]
        results.loc[len(results)] = result
        results.to_csv(outfile_name, index=False)

    #######################################
    # Take component-wise mean of neighbour embeddings
    method = 'mean columns'
    already_run = len(results.query(f'drug == "{test_drug}" & vector_method == "{method}"'))
    if not already_run:
        my_vec = mean_column_value(neighbours_embeds)
        result = [test_drug, method, n_neighbours] + my_vec
        results.loc[len(results)] = result
        results.to_csv(outfile_name, index=False)
