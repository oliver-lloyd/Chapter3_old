import pandas as pd
import numpy as np
import torch
import os
from kge.model import KgeModel
from kge.util.io import load_checkpoint


def SimplEScorer(s_emb, p_emb, o_emb, combine: str):
    n = p_emb.size(0)
    # split left/right
    s_emb_h, s_emb_t = torch.chunk(s_emb, 2)
    p_emb_forward, p_emb_backward = torch.chunk(p_emb, 2)
    o_emb_h, o_emb_t = torch.chunk(o_emb, 2)
    if combine == "spo":
        n = 1  # Added to fix shape error thrown in return line for spo combination
        out1 = (s_emb_h * p_emb_forward * o_emb_t).sum()
        out2 = (s_emb_t * p_emb_backward * o_emb_h).sum()
    elif combine == "sp_":
        out1 = (s_emb_h * p_emb_forward).mm(o_emb_t.transpose(0, 1))
        out2 = (s_emb_t * p_emb_backward).mm(o_emb_h.transpose(0, 1))
    elif combine == "_po":
        out1 = (o_emb_t * p_emb_forward).mm(s_emb_h.transpose(0, 1))
        out2 = (o_emb_h * p_emb_backward).mm(s_emb_t.transpose(0, 1))
    else:
        return super().score_emb(s_emb, p_emb, o_emb, combine)
    return (out1 + out2).view(n, -1) / 2.0


if __name__ == '__main__':
    # Load embeddings
    model_path = '../../../Chapter2/analysis/experiments/selfloops/simple/20230929-111630-simple_selfloops/00021/checkpoint_best.pt'
    checkpoint = load_checkpoint(model_path)
    data_path = '/Users/fu19841/Documents/thesis_analysis/kge/data/selfloops'
    checkpoint['config'].set('dataset.name', data_path)
    model_state = KgeModel.create_from(checkpoint).state_dict()

    node_embeds = model_state['_entity_embedder._embeddings.weight']
    rel_embeds = model_state['_relation_embedder._embeddings.weight']

    # Load estimated vectors
    est_vectors = pd.read_csv('vector_estimates.csv')
    est_vectors = est_vectors.query('num_neighbours > 1')  # Restrict estimates to cases when we have at least 2 neighbours
    est_drugs = est_vectors.drug.unique()

    # Load node info
    node_list = pd.read_csv(
        checkpoint['config'].get('dataset.name') + '/entity_ids.del',
        sep='\t', header=None
    )
    node_name_to_index = {row[1]: row[0] for _, row in node_list.iterrows()}
    node_index_to_name = {row[0]: row[1] for _, row in node_list.iterrows()}
    all_drugs = [node for node in node_list[1] if node.startswith('CID')]

    # Load relation info
    rel_list = pd.read_csv(
        checkpoint['config'].get('dataset.name') + '/relation_ids.del',
        sep='\t', header=None
    )
    rel_name_to_index = {row[1]: row[0] for _, row in rel_list.iterrows()}
    rel_index_to_name = {row[0]: row[1] for _, row in rel_list.iterrows()}
    poly_path = '../../../Chapter2/data/processed/polypharmacy/polypharmacy_edges.tsv'
    poly_edges = pd.read_csv(poly_path, sep='\t', header=None)

    # Sample from real edges to reduce compute
    n_sample = 10000
    poly_edges = poly_edges.loc[np.random.choice(poly_edges.index, n_sample, replace=False)]  # delete me after testing

    # Analyse
    results = []
    for method, subdf in est_vectors.groupby('vector_method'):
        if method != 'actual embeddings':
            for i, row in poly_edges.iterrows():
                head = row[0]
                tail = row[2]
                head_check = head in est_drugs
                tail_check = tail in est_drugs
                if not any([head_check, tail_check]):
                    # Ignore edge if we dont have estimates for either head or tail vectors
                    continue
                else:
                    # Look up actual vectors
                    head_vec = torch.tensor(node_embeds[node_name_to_index[head]])
                    tail_vec = torch.tensor(node_embeds[node_name_to_index[tail]])
                    rel = row[1]
                    rel_vec = torch.tensor(rel_embeds[rel_name_to_index[rel]])

                    # Calculate actual score
                    actual_score = SimplEScorer(head_vec, rel_vec, tail_vec, combine='spo')
                    result = [head, rel, tail, method, actual_score.item()]

                    # Calculate scores with estimated head vector
                    if head_check:
                        query = subdf.query(f'drug == "{head}"')
                        head_vec_est = torch.tensor(query.iloc[0][[str(i) for i in range(256)]])
                        score = SimplEScorer(head_vec_est, rel_vec, tail_vec, combine='spo')
                        result.append(score.item())
                    else:
                        result.append(np.nan)

                    # As above for estimated tail vector
                    if tail_check:
                        query = subdf.query(f'drug == "{tail}"')
                        tail_vec_est = torch.tensor(query.iloc[0][[str(i) for i in range(256)]])
                        score = SimplEScorer(head_vec, rel_vec, tail_vec_est, combine='spo')
                        result.append(score.item())
                    else:
                        result.append(np.nan)

                    if head_check and tail_check:
                        # As above if we have estimates for both vectors
                        score = SimplEScorer(head_vec_est, rel_vec, tail_vec_est, combine='spo')
                        result.append(score.item())
                    else:
                        result.append(np.nan)

                    # Store results
                    results.append(result)

    # Save results. TODO: save per loop so progress not lost if crash
    results = pd.DataFrame(results,
        columns=[
            'head', 'rel', 'tail', 'vec_est_method',
            'actual_score', 'head_est_score', 'tail_est_score', 'both_est_score'
        ]
    )
    results.to_csv(f'score_compare_sample{n_sample}.csv', index=False)
