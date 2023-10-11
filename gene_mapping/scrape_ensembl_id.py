import pandas as pd
import multiprocessing as mp
import requests
from os import listdir
from time import sleep


def scrape_ensembl_id(gene):
    if gene not in listdir('temp'):
        sleep(1)
        base_url = 'https://www.ncbi.nlm.nih.gov/gene/?term='
        response = requests.get(base_url + gene)
        try:
            assert response.status_code == 200
            content = response.text
            ind = content.find('ENSG')
            with open('temp/' + gene, 'w+') as f:
                if ind < 0:
                    # Case when no Ensembl ID found
                    f.write('')
                else:
                    content = content[ind:]
                    ind2 = content.find('"')
                    ensembl = content[:ind2]
                    f.write(ensembl)
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
        pool.map(scrape_ensembl_id, genes)

    out_list = []
    for file_ in listdir('temp'):
        with open('temp/' + file_, 'r') as f:
            ensembl_id = f.read()
        out_list.append([file_, ensembl_id])

    out_df = pd.DataFrame(out_list, columns=['NCBI UID', 'Ensembl ID'])
    out_df.to_csv('NCBI_ensembl_map.csv', index=False)
