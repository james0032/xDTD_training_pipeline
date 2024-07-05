import json
import tqdm

with open("/home/Embeddings/data/drug_disease/disease_ids.json", "r", encoding='utf-8-sig') as disf:
    disj = json.load(disf)
    disj = disj[0]
    disset = set(disj['disease_ids'])
with open("/home/Embeddings/data/drug_disease/drug_ids.json", "r", encoding='utf-8-sig') as drugf:
    drugj = json.load(drugf)
    drugset= set(drugj['drug_ids'])
    
trainedgef = open("/home/Embeddings/data/rkg_embedding_input/train_edges.jsonl", "w")
removededgef2 = open("/home/Embeddings/data/rkg_embedding_input/removed_edges_rev.jsonl", "w")
removededgef = open("/home/Embeddings/data/rkg_embedding_input/removed_edges.jsonl", "w")


with open("/home/Embeddings/data/edges.jsonl") as edgef:
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
           
