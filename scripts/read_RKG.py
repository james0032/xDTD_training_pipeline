# Given a base edges.jsonl from robokop, split drug-disease edges from it to make True Positive, True Negative and unknown drug-disease pairs. The remained robokop edges will then be used for GraphSage input

import random
import argparse
import os
#import jsonlines
import json
from tqdm import tqdm
import pandas as pd
from bmt import Toolkit

tqdm.pandas()

BIOLINK_MODEL_VERSION = os.environ.get("BL_VERSION", "4.1.6")
BIOLINK_MODEL_SCHEMA_URL = f"https://raw.githubusercontent.com/biolink/biolink-model/v{BIOLINK_MODEL_VERSION}/biolink-model.yaml"
PREDICATE_MAP_URL = f"https://raw.githubusercontent.com/biolink/biolink-model/v{BIOLINK_MODEL_VERSION}/predicate_mapping.yaml"

#RKG_ROOT_PATH = "/projects/stars/Data_services/biolink3/graphs/Baseline_Nonredundant/829d9347d0e98317"
pathlist = os.getcwd().split(os.path.sep)
ROOTindex = pathlist.index("xDTD_training_pipeline")
ROOTPath = os.path.sep.join([*pathlist[:(ROOTindex + 1)]])
RKG_ROOT_PATH = os.path.join(ROOTPath, "data")
OUTDIR = os.path.join(RKG_ROOT_PATH,"Split")
if not os.path.exists(OUTDIR):
    os.makedirs(OUTDIR)
    
print(RKG_ROOT_PATH)

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
            if str(sorted(cat_set)) not in self.cat_sets:
                self.cat_sets.add(str(sorted(cat_set)))
                #print(cat_set)
            c+=1
        # Option 1: select one of the leaf category as representatives
        #select_cat = sorted(list(cat_set))[0] 
        # {"{'biolink:Protein', 'biolink:ChemicalEntity'}", 
        # "{'biolink:Drug', 'biolink:SmallMolecule'}", 
        # "{'biolink:MolecularMixture', 'biolink:SmallMolecule'}", 
        # "{'biolink:Protein', 'biolink:MolecularMixture'}", 
        # "{'biolink:Protein', 'biolink:Gene'}", 
        # "{'biolink:Protein', 'biolink:SmallMolecule', 'biolink:Drug'}", 
        # "{'biolink:Protein', 'biolink:Drug'}"}
        
        # Option 2: Concatenate all leaf categories 
        select_cat = ' '.join(sorted([cat.replace("biolink:", "") for cat in cat_set]))
        
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

def all_keep(edge, typemap):
    return False
    
def remove_subclass_and_cid(edge, typemap):
    if edge["predicate"] == "biolink:subclass_of":
        return True
    if edge["subject"].startswith("CAID"):
        return True
    return False

def check_accepted(edge, typemap, accepted):
    subj = edge["subject"]
    obj = edge["object"]
    subj_types = typemap.get(subj, set())
    obj_types = typemap.get(obj, set())
    for acc in accepted:
        if acc[0] in subj_types and acc[1] in obj_types:
            return False
        if acc[1] in subj_types and acc[0] in obj_types:
            return False
    return True

def keep_CD(edge, typemap):
    # return True if you want to filter this edge out
    # We want to keep edges between chemicals and genes, between genes and disease, and between chemicals and diseases
    # Unfortunately this means that we need a type map... Dangit
    if edge["predicate"] == "biolink:subclass_of":
        return True
    accepted = [ ("biolink:ChemicalEntity", "biolink:DiseaseOrPhenotypicFeature") ]
    return check_accepted(edge, typemap, accepted)


def keep_CGD(edge, typemap):
    # return True if you want to filter this edge out
    # We want to keep edges between chemicals and genes, between genes and disease, and between chemicals and diseases
    # Unfortunately this means that we need a type map... Dangit
    if edge["predicate"] == "biolink:subclass_of":
        return True
    accepted = [("biolink:ChemicalEntity", "biolink:Gene"),
                ("biolink:ChemicalEntity", "biolink:DiseaseOrPhenotypicFeature"),
                ("biolink:Gene", "biolink:DiseaseOrPhenotypicFeature")]
    return check_accepted(edge, typemap, accepted)


def keep_CDD(edge, typemap):
    # return True if you want to filter this edge out
    # We want to keep edges between chemicals and genes, between genes and disease, and between chemicals and diseases
    # Unfortunately this means that we need a type map... Dangit
    if edge["predicate"] == "biolink:subclass_of":
        return True
    accepted = [ ("biolink:ChemicalEntity", "biolink:DiseaseOrPhenotypicFeature"),
                ("biolink:DiseaseOrPhenotypicFeature", "biolink:DiseaseOrPhenotypicFeature")]
    return check_accepted(edge, typemap, accepted)

def keep_CCD(edge, typemap):
    # return True if you want to filter this edge out
    # We want to keep edges between chemicals and genes, between genes and disease, and between chemicals and diseases
    # Unfortunately this means that we need a type map... Dangit
    if edge["predicate"] == "biolink:subclass_of":
        return True
    accepted = [ ("biolink:ChemicalEntity", "biolink:DiseaseOrPhenotypicFeature"),
                ("biolink:ChemicalEntity", "biolink:ChemicalEntity")]
    return check_accepted(edge, typemap, accepted)

def keep_CCGDD(edge, typemap):
    # return True if you want to filter this edge out
    # We want to keep edges between chemicals and genes, between genes and disease, and between chemicals and diseases
    # Unfortunately this means that we need a type map... Dangit
    #if edge["predicate"] == "biolink:subclass_of":
    #    return True
    accepted = [ ("biolink:ChemicalEntity", "biolink:DiseaseOrPhenotypicFeature"),
                ("biolink:ChemicalEntity", "biolink:ChemicalEntity"),
                ("biolink:ChemicalEntity", "biolink:Gene"),
                ("biolink:DiseaseOrPhenotypicFeature", "biolink:DiseaseOrPhenotypicFeature"),
                ("biolink:Gene", "biolink:DiseaseOrPhenotypicFeature")]
    return check_accepted(edge, typemap, accepted)

def keep_CGGD(edge, typemap):
    # return True if you want to filter this edge out
    # We want to keep edges between chemicals and genes, between genes and disease, and between chemicals and diseases
    # Unfortunately this means that we need a type map... Dangit
    #if edge["predicate"] == "biolink:subclass_of":
    #    return True
    accepted = [ ("biolink:ChemicalEntity", "biolink:DiseaseOrPhenotypicFeature"),
                ("biolink:ChemicalEntity", "biolink:Gene"),
                ("biolink:Gene", "biolink:Gene"),
                ("biolink:Gene", "biolink:DiseaseOrPhenotypicFeature")]
    return check_accepted(edge, typemap, accepted)

def keep_CCGGDD(edge, typemap):
    # return True if you want to filter this edge out
    # We want to keep edges between chemicals and genes, between genes and disease, and between chemicals and diseases
    # Unfortunately this means that we need a type map... Dangit
    #if edge["predicate"] == "biolink:subclass_of":
    #    return True
    accepted = [ ("biolink:ChemicalEntity", "biolink:DiseaseOrPhenotypicFeature"),
                ("biolink:ChemicalEntity", "biolink:ChemicalEntity"),
                ("biolink:ChemicalEntity", "biolink:Gene"),
                ("biolink:Gene", "biolink:Gene"),
                ("biolink:Gene", "biolink:DiseaseOrPhenotypicFeature"),
                ("biolink:DiseaseOrPhenotypicFeature", "biolink:DiseaseOrPhenotypicFeature"),]
    return check_accepted(edge, typemap, accepted)


def pred_trans(edge, edge_map):
    edge_key = {"predicate": edge["predicate"]}
    edge_key["subject_aspect_qualifier"] = edge.get("subject_aspect_qualifier", "")
    edge_key["object_aspect_qualifier"] = edge.get("object_aspect_qualifier", "")
    edge_key["subject_direction_qualifier"] = edge.get("subject_direction_qualifier", "")
    edge_key["object_direction_qualifier"] = edge.get("object_direction_qualifier", "")
    edge_key_string = json.dumps(edge_key, sort_keys=True)
    if edge_key_string not in edge_map:
        edge_map[edge_key_string] = f"predicate:{len(edge_map)}"
    return edge_map[edge_key_string]


def dump_edge_map(edge_map, outdir):
    output_file=f"{outdir}/edge_map.json"
    with open(output_file, "w") as writer:
        json.dump(edge_map, writer, indent=2)

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
        
def class_label(x, tpp, tnp, treat, contra):
    groupp = set(x.predicate.values)
    boolp = any(i in tpp for i in groupp)
    booln = any(i in tnp for i in groupp)
    
    if boolp and booln:
        x['y'] = 9       # 9: predicate conflicted
    elif boolp and not booln:
        if any(i in treat for i in groupp):
            x['y'] = 1   # 1: treat edges (true positives)
        else:
            x['y'] = 2   # 2: relax def of treat predicates
    elif not boolp and booln:
        if any(i in contra for i in groupp):
            x['y'] = 0   # 0: contra indicated edges (true negatives)
        else:
            x['y'] = -1  #-1: relax def of not treat predicates
    return x

def split_ddpair_dump(df, drug_ids, disease_ids, treat, contra, method="strict", n_random=None):
        
    dftp = df[df['predicate'].isin(treat)].reset_index(drop=True)
    dftn = df[df['predicate'].isin(contra)].reset_index(drop=True)
    dftp.to_csv(os.path.join(RKG_ROOT_PATH, "tp_pairs_allColumns.txt"), sep='\t', index=None)
    dftn.to_csv(os.path.join(RKG_ROOT_PATH, "tn_pairs_allColumns.txt"), sep='\t', index=None)
    dftp[['subject', 'object']].rename(columns={'subject':'source', 'object':'target'}).to_csv(os.path.join(RKG_ROOT_PATH, "tp_pairs.txt"), sep='\t', index=None)
    dftn[['subject', 'object']].rename(columns={'subject':'source', 'object':'target'}).to_csv(os.path.join(RKG_ROOT_PATH, "tn_pairs.txt"), sep='\t', index=None)
    
    if n_random: # create 500 random pairs to match RTX group did 
        print(f"Randomly generating {n_random*2} pairs for each drug-disease treat edge.")
        random_pairs = []
        for drug in tqdm(set(dftp['subject'])):
            exists = []
            for disease in random.sample(disease_ids, 500):
                this_pair = drug+disease
                if this_pair in df['string']:
                    exists.append(this_pair)
                    continue
                random_pairs.append({"source": drug,
                                     "target": disease,
                                     "y": 3
                
                })
            #if len(exists) >0:
            #    print(f"drug-disease pairs exist: {exists}")
        for disease in tqdm(set(dftp['object'])):
            exists = []
            for drug in random.sample(drug_ids, 500):
                this_pair = drug+disease
                if this_pair in df['string']:
                    exists.append(this_pair)
                    continue
                random_pairs.append({"source": drug,
                                     "target": disease,
                                     "y": 3
                
                })
            #if len(exists) >0:
            #    print(f"drug-disease pairs exist: {exists}")
                      
        dfrandp = pd.DataFrame(random_pairs)
        dfrandp.to_csv(os.path.join(RKG_ROOT_PATH, "random_pairs.txt"), sep='\t', index=None)
    #    exist_pairs = set(dfpairs['string'].values)
    #    for drug in set(dffinal['subject']):
    #        for n in disease_ids:
                
        
def id_collector(node_file):
    drug_ids = []
    drug_cat = ["biolink:SmallMolecule", "biolink:Drug"]
    disease_ids = []
    disease_cat = ["biolink:DiseaseOrPhenotypicFeature", ]
    type_map = {}
    with open(node_file) as reader:
        for i, l in enumerate(tqdm(reader)):
            node = json.loads(l)
            node_cats = set(node['category'])
            type_map[node['id']] = node_cats
            if drug_cat[0] in node_cats or drug_cat[1] in node_cats:
                drug_ids.append(node['id'])
            elif disease_cat[0] in node_cats:
                disease_ids.append(node['id'])
    drug_ids = set(drug_ids)
    disease_ids = set(disease_ids)
    print(f"There are {len(drug_ids)} unique drugs/small molecules in the graph.")
    print(f"There are {len(disease_ids)} unique disease or phyenotypic features in the graph.")
    return drug_ids, disease_ids, type_map
    
def create_start_graph(node_file=os.path.join(RKG_ROOT_PATH,"nodes.jsonl"), edges_file=os.path.join(RKG_ROOT_PATH,"edges.jsonl"), style="nonredundant"):
    
    print("Collect all existing drugs and disease lists.")
    drug_ids, disease_ids, type_map = id_collector(node_file)
    
    # Separate drug-disease pairs from original graph
    print("Separate drug-disease pairs from original graph")
    with open(os.path.join(OUTDIR,"train_edges.jsonl"), "w") as trainedges:
        with open(os.path.join(OUTDIR,"all_drug_disease_pairs_edges.jsonl"), "w") as ddpair_edges:
            with open(os.path.join(OUTDIR,"drug_disease_pairs_rev_edges.jsonl"), "w") as revedges:
                with open(edges_file) as reader:
                    for i, l in enumerate(tqdm(reader)):
                        edge = json.loads(l)
                        if (edge['subject'] in drug_ids) and (edge['object'] in disease_ids):
                            ddpair_edges.write(l)
                        elif (edge['subject'] in disease_ids) and (edge['object'] in drug_ids):
                            revedges.write(l)
                        else:
                            trainedges.write(l)
                            
    outdir = os.path.join(RKG_ROOT_PATH, style)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    output_file = f"{outdir}/rotorobo.txt"
    
    
    if style == "rmsubclass":
        remove_edge = remove_subclass_and_cid
    elif style == "nonredundant":
        remove_edge = all_keep
    elif style == "keep_CCGGDD":
        remove_edge = keep_CCGGDD
        
    print(f"Edge file formatting using style {style}.")    
    df_edges = []
    c =0
    edge_map = {}
    with open(os.path.join(OUTDIR,"train_edges.jsonl"), "r") as edgef:
        with open(output_file, "w") as writer:
            for i,l in enumerate(tqdm(edgef)):
                edge = json.loads(l)
                if remove_edge(edge, type_map):
                    c+=1
                    continue
                df_edges.append(format_edge(edge))
                writer.write(f"{edge['subject']}\t{pred_trans(edge,edge_map)}\t{edge['object']}\n")
                
    
    dump_edge_map(edge_map,outdir)
    print("edge map dump completed.")
    
    df_edges = pd.DataFrame(df_edges) 
    edge_format = os.path.join(RKG_ROOT_PATH,"filtered_graph_edges.txt")
    df_edges.to_csv(edge_format, sep='\t', index=False)
    print(f"formatted edge file dump for embedding completed.\n{edge_format}")
    # Keep all nodes instead of removing orphan nodes; still has its own script section for adding filter back to orphan nodes instead of combine to the first node file read.
    print("node file formatting")
    sset = set(df_edges['source'])
    tset = set(df_edges['target'])
    edge_set = sset.union(tset)
    blt = bltools()
    df_nodes = []
    with open(node_file, "r") as nodef:
    #with open("/Users/jchung/Documents/DOCKER/miniAIxB/Embeddings/data/nodes.jsonl", "r") as nodef:
        for i, l in enumerate(tqdm(nodef)):
            j = json.loads(l)
            if j['id'] in edge_set:
                dic, c =  blt.format_node(j, c)
                df_nodes.append(dic)

            #else:
            #    orphanf.write(l)


    #print(blt.cat_sets)       
    df_nodes = pd.DataFrame(df_nodes)
    node_format = os.path.join(RKG_ROOT_PATH, "filtered_graph_nodes_info.txt")
    df_nodes.to_csv(node_format, sep='\t', index=False)  
    print(f"formatted node file dump for embedding completed.\n{node_format}")
    
    # Formatting train and test sets from drug-disease pairs
    print("Labeling drug-disease pairs")
    treat = set(['biolink:treats'])
    contra = set(['biolink:contraindicated_in'])
    tppredicates = set(['biolink:ameliorates_condition', 'biolink:treats', 'biolink:treats_or_applied_or_studied_to_treat', 'biolink:preventative_for_condition'])
    tnpredicates = set(['biolink:causes', 'biolink:contraindicated_in', 'biolink:contributes_to'])
    
    dfpairs = pd.read_json(os.path.join(OUTDIR,"all_drug_disease_pairs_edges.jsonl"), lines=True)
    dfpairs['string'] = dfpairs['subject'] + dfpairs['object']
    labels = []
    exists = dict()
    # multiple edges for one drug-disease pair.
    # If conflict, leave them behind for now. Label as existing edges with other predicates (9)
    print("Groupby class labeling.")
    dfpairs = dfpairs.groupby(['subject','object']).progress_apply(class_label, tpp=tppredicates, tnp=tnpredicates, treat=treat, contra=contra).reset_index(drop=True)
    dfpairs.to_csv(os.path.join(OUTDIR, "all_drug_disease_pairs_edges.tsv"), sep='\t', index=None)
    print("drug-disease pair dump.")
    split_ddpair_dump(dfpairs, drug_ids, disease_ids, treat, contra, n_random=500)
    
    
    
    
if __name__ == "__main__":
    print(RKG_ROOT_PATH)
    parser = argparse.ArgumentParser()
    parser.add_argument("--style", type=str, help="How to select subset of knowledge graph", default="nonredundant")
    args = parser.parse_args()
    create_start_graph(style=args.style)
    
    