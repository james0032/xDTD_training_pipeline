import joblib
import os
import sys
import pandas as pd
import numpy as np
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
def find_key(ID, eIDs, nodelist):
    #ID = x["single_ID"]
    #eIDs = x["Equivalent_IDs"]
    if ID in nodelist:
        print(f"found {ID} on single_ID column")
        return ID
    else:
        keys = eIDs.split("|")[0].replace("[", "").replace("]", "").replace("\"", "").replace("\'", "").replace(" ", "").split(",")
        for k in keys:
            if k in nodelist:
                print(f"found {k} on Equivalent_IDs column")
                return k
        print(f"Used KG does not have ID {keys}")    
        return KEYNOTEXIST

print(f"Number of embedded nodes in this embedding layer is {len(bioemd_dict)}")
dfdrug = pd.read_csv(os.path.join(ROOTPATH, "drug_list/v110/matrix-drug-list-1.1.0/drug-list/data/03_primary/drugList.tsv"), sep='\t', header=0)
dfdrug = dfdrug.drop_duplicates(subset=["single_ID"]).reset_index(drop=True)
dfdrug["found_ID"] = dfdrug.apply(lambda x: find_key(x.single_ID, x.Equivalent_IDs, bioemd_dict.keys()), axis=1)
dfdrug = dfdrug[-(dfdrug["found_ID"]==KEYNOTEXIST)].reset_index(drop=True)
dfdrug.to_csv(os.path.join(ROOTPATH, "drug_list/v110/matrix-drug-list-1.1.0/drug-list/data/03_primary/drugList_KeyNotExist.tsv"))
print("After keynotexist removed")
print(dfdrug[dfdrug["found_ID"]==KEYNOTEXIST]["single_ID"].values)
print(dfdrug.shape)

dfind = pd.read_csv(os.path.join(ROOTPATH, "dis_list/matrix-disease-list-2024-10-08/matrix-disease-list.tsv"), sep='\t', header=0)
dfind["in_keys"] = dfind["category_class"].isin(bioemd_dict.keys())
print(dfind["in_keys"].value_counts())
dfind = dfind[dfind["in_keys"]].reset_index(drop=True)
print(dfind.shape)

dfdrug["emb_vector"] = dfdrug["found_ID"].apply(lambda x: bioemd_dict[x])
dfind["emb_vector"] = dfind["category_class"].apply(lambda x: bioemd_dict[x])
testcase = dfdrug["emb_vector"][1]
print(f"Example emb vector and size {testcase}, {len(testcase)}")

def generate_X(drug_vs, disv):
    disv = np.tile(disv, [len(drug_vs), 1])
    drug_vs = np.stack(drug_vs)
    return np.concatenate((drug_vs, disv), axis=1)

dfcur = dfdrug[["single_ID", "ID_Label"]].rename(columns={"single_ID": "Drug_ID", "ID_Label": "Drug_Name"})
for idx, row in dfdrug.head(10).iterrows():
    dfcur["Disease_ID"] = row["category_class"] # disease ID
    dfcur["Disease_Name"] = row["label"] # disease name
    cur_result = fitModel.predict_proba(generate_X(dfdrug["emb_vector"], row["emb_vector"]))
    dfcur["probability"] = cur_result[:,0]
    dfcur["prediction"] = cur_result[:,1]
    
    dfcur.to_csv(os.path.join(ROOTPATH, f"results/graphsage/gs_emb_prediction_{row["category_class"]}.csv"), index=False, header=False)