import joblib
import os
import sys
import pandas as pd
import pickle

## Import Personal Packages
pathlist = os.getcwd().split(os.path.sep)
ROOTindex = pathlist.index("xDTD_training_pipeline")
ddpath = os.path.sep.join([*pathlist[:(ROOTindex + 1)]])
sys.path.append(os.path.join(ddpath, 'scripts'))
import utils

emb_name = "graphsage"; tpstyle="stringent"; tnstyle="stringent"
model_name = f'RF_model_{emb_name}_{tpstyle}_{tnstyle}.pt'
fitModel = joblib.load(os.path.join(ddpath, model_name))

if emb_name == "biobert":
    # biobert embeddings
    print("Read Biobert embedding layer.")
    with open(f"{os.path.join(ddpath, 'data/text_embedding/embedding_biobert_namecat.pkl')}", "rb") as infile:
        bioemd_dict = pickle.load(infile)
elif emb_name == "graphsage":
# Graphsage output embeddings
    print("Read graphsage embedding layer.")
    with open(f"{os.path.join(ddpath, 'data/graphsage_output/unsuprvised_graphsage_entity_embeddings.pkl')}", "rb") as infile:
        bioemd_dict = pickle.load(infile)

ROOTPATH = "/projects/aixb/jchung/everycure/alltoall"

dfdrug = pd.read_csv(os.path.join(ROOTPATH, "drug_list/v104", "drugList.tsv"), sep='\t', index_col=0)
dfdrug = dfdrug.drop_duplicates(subset=["single_ID"])
dfind = pd.read_csv(os.path.join(ROOTPATH, "dis_list/v2408", "matrix-disease-list.tsv"), sep='\t')
    
print(list(emb_name.keys())[:10])