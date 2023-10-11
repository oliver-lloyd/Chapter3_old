import pandas as pd
import multiprocessing as mp
import requests


def scrape_ensembl_id(gene):
    base_url = 'https://www.ncbi.nlm.nih.gov/gene/?term='
    response = requests.get(base_url + gene)
    try:
        assert response.status_code == 200
        content = response.text
        ind = content.find('ENSG')
        if ind < 0:
            return [gene, None]
        else:
            content = content[ind:]
            ind2 = content.find('"')
            ensembl = content[:ind2]
            return [gene, ensembl]
    except AssertionError:
        print(f'Gene: "{gene}" returned response code: {response.status_code}. Exiting..')
        quit()


if __name__ == '__main__':

    # Load entity list
    ents = pd.read_csv('../entity_ids.del', sep='\t', index_col=0, header=None)

    # Extract genes
    genes = []
    for id_ in ents[1]:
        try:
            int(id_)
            genes.append(id_)
        except ValueError:
            continue
    del ents

    with mp.Pool(mp.cpu_count()) as pool:
        mappings = pool.map(scrape_ensembl_id, genes)

    out_df = pd.DataFrame(mappings, columns=['NCBI UID', 'Ensembl ID'])
    out_df.to_csv('gene_to_ensembl.csv', index=False)
