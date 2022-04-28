from .gspan_mining.config import parser
from .gspan_mining.main import main
import json
import glob
"""
datasets = glob.glob("mydata/egos_4/*.txt")
#print(datasets)
exeptions = ["mydata/egos_4/ba2_5labels_egos.txt", "mydata/egos_4/ba2_15labels_egos.txt"]

dataset_results = []
cnt = 0
l = len(datasets)


for dataset in list(filter(lambda x: x not in exeptions, datasets)):
    print(dataset)
    min_support = 100
    '''    if "BBBP" in dataset:
        min_support = 130 // 2
    if dataset == 'mydata/egos2/egos_2/mutag_33labels_egos.txt':
        print("!!!")'''
    args_str = f'-s {min_support} -p False -v True -mu 0.000001 -U True -k 10 {dataset}'
    FLAGS, _ = parser.parse_known_args(args=args_str.split())
    gs = main(FLAGS)
    dataset_results.append(gs.best_pattern)
    cnt += 1
    print(f"{cnt} out of {l} is done.")
with open("results.json", 'w+') as json_file:
    json_file.write(json.dumps(dataset_results))

"""

def run_dset(dataset, pattern):
    prefix = "results/egos/"
    filename = prefix+dataset+"_"+str(pattern)+"labels_egos.txt"
    min_support = 100

    args_str = f'-s {min_support} -p False -v True -mu 0.000001 -U True -k 10 {filename}'
    FLAGS, _ = parser.parse_known_args(args=args_str.split())
    gs = main(FLAGS)
    dataset_results = gs.best_pattern

    return dataset_results
