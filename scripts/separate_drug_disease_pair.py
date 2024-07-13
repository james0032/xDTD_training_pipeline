import json
import tqdm
import os

with open("/home/jchung/ROBOKOP/git/xDTD_training_pipeline/data/drug_disease/disease_ids.json", "r", encoding='utf-8-sig') as disf:
    disj = json.load(disf)
    disj = disj[0]
    disset = set(disj['disease_ids'])
with open("/home/jchung/ROBOKOP/git/xDTD_training_pipeline/data/drug_disease/drug_ids.json", "r", encoding='utf-8-sig') as drugf:
    drugj = json.load(drugf)
    drugset= set(drugj['drug_ids'])

ROOTPATH = "/home/jchung/ROBOKOP/git/xDTD_training_pipeline"
rkgf = os.path.join(ROOTPATH, "data/rkg_embedding_input")
if not os.path.exists(rkgf): 
    os.makedirs(rkgf)
trainedgef = open(os.path.join(rkgf, "train_edges.jsonl"), "w")
removededgef2 = open(os.path.join(rkgf, "removed_edges_rev.jsonl"), "w")
removededgef = open(os.path.join(rkgf, "removed_edges.jsonl"), "w")


with open(os.path.join(ROOTPATH, "data/edges.jsonl")) as edgef:
    for i, l in enumerate(tqdm.tqdm(edgef)):
        j = json.loads(l)
        if (j['subject'] in drugset and j['object'] in disset):
            removededgef.write(l)
        elif (j['object'] in drugset and j['subject'] in disset):
            removededgef2.write(l)
        else:
            trainedgef.write(l)
            
        #if i > 1000000:
        #    break        
        
 
trainedgef.close()
removededgef.close()
removededgef2.close()
           
