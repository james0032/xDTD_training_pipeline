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

emb_name = "graphsage"; 
tpstyle="stringent"; 
tnstyle="stringent"

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
KEYNOTEXIST = "KeyNotExist"
def find_key(x, nodelist):
    ID = x["single_ID"]
    eIDs = x["Equivalent_IDs"]
    if ID in nodelist:
        return ID
    else:
        keys = eIDs.split("|")[0].replace("[", "").replace("]", "").replace("\"", "").replace("\'", "").replace(" ", "").split(",")
        for k in keys:
            if k in nodelist:
                return k
            
        return KEYNOTEXIST


dfdrug = pd.read_csv(os.path.join(ROOTPATH, "drug_list/v110/matrix-drug-list-1.1.0/drug-list/data/03_primary/drugList.tsv"), sep='\t', header=0)
dfdrug = dfdrug.drop_duplicates(subset=["single_ID"]).reset_index(drop=True)
dfdrug["found_ID"] = dfdrug.apply(lambda x: find_key(x, bioemd_dict.keys()))
dfind = pd.read_csv(os.path.join(ROOTPATH, "dis_list/matrix-disease-list-2024-10-08/matrix-disease-list.tsv"), sep='\t', header=0)
    
#dfdrug["in_keys"] = dfdrug["single_ID"].isin(bioemd_dict.keys())
print(f"Number of embedded nodes in this embedding layer is {len(bioemd_dict)}")
print(dfdrug["found_ID"].value_counts())
print(dfdrug[dfdrug["found_ID"]==KEYNOTEXIST]["single_ID"].values)

dfind["in_keys"] = dfind["category_class"].isin(bioemd_dict.keys())

print(dfind["in_keys"].value_counts())
dfind = dfind[dfind["in_keys"]].reset_index(drop=True)
print(dfind.shape)

