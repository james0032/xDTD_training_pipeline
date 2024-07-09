import json
import pandas as pd
from tqdm import tqdm

def format_edge(j):
    if 'publications' in j:
            npub = int(len(j['publications']))
    else:
        npub=0
    dic = {"source": j['subject'],
                         "target": j['object'],
                         "predicate": j['predicate'],
                         "num_publications": npub,
                         "p_knowledge_source": j['primary_knowledge_source']}
    return dic

#with open("/home/jchung/ROBOKOP/git/xDTD_training_pipeline/data/rkg_embedding_input/filtered_graph_edges.txt", "w") as newedgef:        
df_edges = []
with open("/home/jchung/ROBOKOP/git/xDTD_training_pipeline/data/rkg_embedding_input/train_edges.jsonl", "r") as edgef:
    for i,l in enumerate(tqdm(edgef)):
        j = json.loads(l)
        if j['predicate']!='biolink:subclass_of':
            df_edges.append(format_edge(j))
            
df_edges = pd.DataFrame(df_edges) 
df_edges.to_csv("/home/jchung/ROBOKOP/git/xDTD_training_pipeline/data/rkg_embedding_input/filtered_graph_edges.txt", sep='\t', index=False)            
            
