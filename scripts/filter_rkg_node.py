import json
import pandas as pd
from tqdm import tqdm
from bmt import Toolkit
import os

BIOLINK_MODEL_VERSION = os.environ.get("BL_VERSION", "4.1.6")
BIOLINK_MODEL_SCHEMA_URL = f"https://raw.githubusercontent.com/biolink/biolink-model/v{BIOLINK_MODEL_VERSION}/biolink-model.yaml"
PREDICATE_MAP_URL = f"https://raw.githubusercontent.com/biolink/biolink-model/v{BIOLINK_MODEL_VERSION}/predicate_mapping.yaml"

class bltools():
    def __init__(self):
        self.bmt = self.get_biolink_model_toolkit()
        self.cat_sets = set()

    def get_biolink_model_toolkit(self):
        return Toolkit(schema=BIOLINK_MODEL_SCHEMA_URL, predicate_map=PREDICATE_MAP_URL)

    def find_biolink_leaves(self, biolink_concepts: set):
        """
        Given a list of biolink concepts, returns the leaves removing any parent concepts.
        :param biolink_concepts: list of biolink concepts
        :return: leave concepts.
        """
        ancestry_set = set()  # the set of concepts that are parents to concepts in the set
        unknown_elements = set()  # concepts not found in the biolink model
        for x in biolink_concepts:
            current_element = self.bmt.get_element(x)
            if not current_element:
                unknown_elements.add(x)
            ancestors = set(self.bmt.get_ancestors(x, mixin=True, reflexive=False, formatted=True))
            ancestry_set = ancestry_set.union(ancestors)
        leaf_set = biolink_concepts - ancestry_set - unknown_elements
        return leaf_set

    def format_node(self, j, c):
        
        # Marked and substitued by find_biolink_leaves
        """ 
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
        """
        
        cat_set = self.find_biolink_leaves(set(j['category']))
        if len(cat_set) > 1:
            if str(cat_set) not in self.cat_sets:
                self.cat_sets.add(str(cat_set))
                #print(cat_set)
            c+=1
        select_cat = sorted(list(cat_set))[0]
        # {"{'biolink:Protein', 'biolink:ChemicalEntity'}", 
        # "{'biolink:Drug', 'biolink:SmallMolecule'}", 
        # "{'biolink:MolecularMixture', 'biolink:SmallMolecule'}", 
        # "{'biolink:Protein', 'biolink:MolecularMixture'}", 
        # "{'biolink:Protein', 'biolink:Gene'}", 
        # "{'biolink:Protein', 'biolink:SmallMolecule', 'biolink:Drug'}", 
        # "{'biolink:Protein', 'biolink:Drug'}"}
        
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
        return dic, c

print("Get connected edge ids.")
df_edges = pd.read_csv("/home/jchung/ROBOKOP/git/xDTD_training_pipeline/data/filtered_graph_edges.txt", sep="\t", header=0)
sset = set(df_edges['source'])
tset = set(df_edges['target'])
edge_set = sset.union(tset)
blt = bltools()


df_nodes = []
orphanf = open("/home/jchung/ROBOKOP/git/xDTD_training_pipeline/data/orphan_nodes.jsonl", "w")
with open("/home/jchung/ROBOKOP/git/xDTD_training_pipeline/data/nodes.jsonl", "r") as nodef:
#with open("/Users/jchung/Documents/DOCKER/miniAIxB/Embeddings/data/nodes.jsonl", "r") as nodef:
    for i, l in enumerate(tqdm(nodef)):
        j = json.loads(l)
        if j['id'] in edge_set:
            dic, c =  blt.format_node(j, c)
            df_nodes.append(dic)
            
        else:
            orphanf.write(l)


print(blt.cat_sets)       
df_nodes = pd.DataFrame(df_nodes)
df_nodes.to_csv("/home/jchung/ROBOKOP/git/xDTD_training_pipeline/data/filtered_graph_nodes_info.txt", sep='\t', index=False)    

orphanf.close()