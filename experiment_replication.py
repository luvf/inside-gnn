
from ExplanationEvaluation.configs.selector import Selector
from ExplanationEvaluation.tasks.replication import replication
from ExplanationEvaluation.evaluation.AUCEvaluation import AUCEvaluation
from ExplanationEvaluation.evaluation.PREvaluation import PREvaluation

from ExplanationEvaluation.evaluation.FidelityEvaluation import FidelityEvaluation
from ExplanationEvaluation.evaluation.InFidelityEvaluation import InFidelityEvaluation

from ExplanationEvaluation.evaluation.SparsityEvaluation import SparsityEvaluation


def run_xp(_dataset,_explainer, metrics, **kwargs ):
    _folder = 'replication' # One of: replication, extension

    config_path = f"./ExplanationEvaluation/configs/{_folder}/explainers/{_explainer}/{_dataset}.json"

    config = Selector(config_path)
    extension = (_folder == 'extension')

    return replication(config.args.explainer, extension,run_qual=False, metrics=metrics,  **kwargs)

def latex_metrics(data, name):
    print(name)
    l = str()
    for x,v in data.items():
        l += x+ " "
        for dname, vals in v.items():
            if name == "sparsity" or name == "time":
                l+= " & "+ format(vals[0][name],'.3f')
            else :
                for x in list(vals[0][name]):
                    l+= " & "+ format(x,'.3f')
        l+="\\\\\n"
    print(l)
#%%
gt_dsets = { "dd", "bacommunity", "treecycles", "treegrids", "ba2motifs", "mutag"}
datasets = ["dd","proteins", "ba2motifs", "aids", "bbbp", "mutag"]
datasets = ["proteins" ]

#datasets = ["aids", "bbbp", "ba2motifs", "mutag"]

#datasets = ["ba2motifs"]
savepoint = {
    "rerun_extraction": True,
    "rerun_pattern_mining" : False,
    "rerun_perturbation_serch": False
}

experiments = [
    #{"xpl":"svxexplainer","name": "svxexplainer"},
    #{"xpl": "actexplainer", "topK": False, "K": 0, "policy_name": "ego", "name": "Inside GNN ego", "k": 10, "motifs": "base", "negation": False},
    #{"xpl": "actexplainer", "topK": False, "K": 0, "policy_name": "node", "name": "Inside GNN ode", "k": 10, "motifs": "base", "negation": False},
    #{"xpl": "actexplainer", "topK": False, "K": 0, "policy_name": "decay", "name": "Inside GNN decay", "motifs": "base", "negation": False},
    #{"xpl": "actexplainer", "topK": False, "K": 0, "policy_name": "top", "name": "Inside GNN top 5", "k": 5, "motifs": "base", "negation": False},
    #{"xpl": "actexplainer", "topK": False, "K": 0, "policy_name": "top", "name": "Inside GNN top 10", "k": 10, "motifs": "base", "negation": False},

    # {"xpl": "gradexplainer", "topK": False, "K": 0, "policy_name": "top", "name": "grad_explainer", "k": 10,
    # "pertub_mode": "mean", "motifs": "base"},

    #{"xpl":"gnnexplainer","name": "GnnExplainer"},

    #{"xpl":"pgexplainer","name": "PGExplainer"},
    #{"xpl": "pgmexplainer", "topK": False, "K": 0, "policy_name": "top", "name": "pgmexplainer", "k": 10, "pertub_mode": "mean", "motifs": "base"}
    {'xpl': 'actexplainer', 'topK': False, 'K': 0, 'policy_name': 'ego', 'name': 'Inside GNN ego', 'k': 10, 'motifs': 'base', 'rerun_extraction': True, 'rerun_pattern_mining': True, 'rerun_perturbation_serch': True, 'groundtruth': False}]





metrics ={
    #"auc" : [AUCEvaluation,dict()],
    #"pr" : [PREvaluation , dict()],
    "fid": [FidelityEvaluation, {"topK": False}],
    "infid" :[InFidelityEvaluation, {"topK": False}],
    #"infid 5" :[InFidelityEvaluation, {"topK": True, "K": 5}],
    #"infid 10" :[InFidelityEvaluation, {"topK": True, "K": 10}],

    #"fid 5" : [FidelityEvaluation, {"topK":True, "K":5}],
    #"fid 10": [FidelityEvaluation, {"topK": True, "K": 10}],

    "sparsity": [SparsityEvaluation, dict()],

}

metrics_tex = ["fid","fid 5","fid 10" ,"infid", "infid 5", "infid 10", "sparsity", "time"]
metrics_tex = ["fid" ,"infid", "sparsity", "time"]

out = dict()

def run_experiments(experiments, metrics, metrics_tex,savepoint, gt_datasets):
    for xp in experiments:
        xp.update(savepoint)
        print(xp)
        out[xp["name"]] = dict()
        for d in datasets:
            xp["groundtruth"]= d in gt_dsets
            res= run_xp(d, xp["xpl"], metrics,  **xp)
            print(res)
            out[xp["name"]][d]= res

            [latex_metrics(out, metric) for metric in metrics_tex]
    [latex_metrics(out, metric) for metric in metrics_tex ]


run_experiments(experiments, metrics, metrics_tex, savepoint,gt_dsets)