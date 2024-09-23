import sys
import os
import pandas as pd
from read_RKG import id_collector, split_ddpair_dump

pathlist = os.getcwd().split(os.path.sep)
ROOTindex = pathlist.index("xDTD_training_pipeline")
ROOTPath = os.path.sep.join([*pathlist[:(ROOTindex + 1)]])
print(f"rootpath is {ROOTPath}")
sys.path.append(os.path.join(ROOTPath, 'scripts'))

RKG_ROOT_PATH = os.path.join(ROOTPath, "data")
OUTDIR = os.path.join(RKG_ROOT_PATH,"Split")
if not os.path.exists(OUTDIR):
    os.makedirs(OUTDIR)
    
print(RKG_ROOT_PATH)

drug_ids, disease_ids, type_map = id_collector(os.path.join(RKG_ROOT_PATH,"nodes.jsonl"))

treat = set(['biolink:treats'])
contra = set(['biolink:contraindicated_in'])
tppredicates = set(['biolink:ameliorates_condition', 'biolink:treats', 'biolink:treats_or_applied_or_studied_to_treat', 'biolink:preventative_for_condition'])
tnpredicates = set(['biolink:causes', 'biolink:contraindicated_in', 'biolink:contributes_to'])

dfpairs = pd.read_json(os.path.join(OUTDIR,"all_drug_disease_pairs_edges.jsonl"), lines=True)
dfpairs['string'] = dfpairs['subject'] + dfpairs['object']

split_ddpair_dump(dfpairs, drug_ids, disease_ids, tppredicates, tnpredicates, n_random=50)