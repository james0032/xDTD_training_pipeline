import json
import pandas as pd
from tqdm import tqdm

def format_node(j):
    if 'biolink:Drug' in j['category']:
        select_cat = 'biolink:Drug'
    elif 'biolink:SmallMolecule' in j['category']:
        select_cat = 'biolink:SmallMolecule'
    else:
        select_cat = j['category'][0]
        
    if 'biolink:DiseaseOrPhenotypicFeature' in j['category']:
        select_cat = 'biolink:DiseaseOrPhenotypicFeature'
    else:
        select_cat = j['category'][0]
    
        #print(j['category'])
    # Combine description and mrdef    
    des = ''
    if 'description' in j:
        des = des + j['description']
    if 'mrdef' in j:
        des = des + j['mrdef']
        
    dic = {"id": j['id'], 
           "category": select_cat,
           "name": j['name'],
           "all_names": j['name'], # temporary, did not really fetch all names
           "des": des
           }
    return dic

df_nodes = []
with open("/home/jchung/ROBOKOP/git/xDTD_training_pipeline/data/raw_graph/nodes.jsonl", "r") as nodef:
    for i, l in enumerate(tqdm(nodef)):
        j = json.loads(l)
        df_nodes.append(format_node(j))
                
                
df_nodes = pd.DataFrame(df_nodes)
df_nodes.to_csv("/home/jchung/ROBOKOP/git/xDTD_training_pipeline/data/rkg_embedding_input/filtered_graph_nodes_info.txt", sep='\t', index=False)    
