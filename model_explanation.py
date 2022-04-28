import torch
import pandas as pd
from tqdm import tqdm
from ExplanationEvaluation.datasets.dataset_loaders import load_dataset
from ExplanationEvaluation.gspan_mine.run import run_dset
import pysubgroup as ps
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from ExplanationEvaluation.explainers.InsideGNN import InsideGNN
from ExplanationEvaluation.tasks.replication import to_torch_graph
from matplotlib.pyplot import figure
import numpy as np
import networkx as nx

import matplotlib
import matplotlib.pyplot as plt
import json
from ExplanationEvaluation.models.model_selector import model_selector

from ExplanationEvaluation.configs.selector import Selector
from ExplanationEvaluation.tasks.replication import replication
from ExplanationEvaluation.evaluation.AUCEvaluation import AUCEvaluation
from ExplanationEvaluation.evaluation.PREvaluation import PREvaluation

from ExplanationEvaluation.evaluation.FidelityEvaluation import FidelityEvaluation
from ExplanationEvaluation.evaluation.InFidelityEvaluation import InFidelityEvaluation

from ExplanationEvaluation.evaluation.SparsityEvaluation import SparsityEvaluation

from ExplanationEvaluation.explainers.utils import get_edge_distribution
from subgraph_metrics import *

def load_activations(dataset_name):
    model = "GNN"
    explainer = "ACT"
    file = 'results/trace/' + dataset_name + "_" + model + "_" + explainer + "_activation.csv"
    df = pd.read_csv(file)
    return df


def load_motifs(dataset):
    names = {"ba2": ("ba2"),
             "aids": ("Aids"),
             "BBBP": ("Bbbp"),
             "mutag": ("Mutag"),
             "DD": ("DD"),
             "PROTEINS_full": ("Proteins")}
    name = names[dataset]
    file = "ExplanationEvaluation/datasets/activations/" + name + "/" + name + "_activation_encode_motifs.csv"
    rules = list()
    with open(file, "r") as f:
        for l in f:
            r = l.split("=")[1].split(" \n")[0]
            label = int(l.split(" ")[3].split(":")[1])
            rules.append((label, r))
    return rules


def get_covers(df, rule):
    r2 = rule[1].split(" ")
    # r self.rs[18]
    # r = list(map((lambda r: r if r[:3]!="l_3" else  "a"+r[4:]), r2))
    r = r2
    rule = ps.Conjunction([ps.EqualitySelector(el, 1) for el in r])
    return df[rule.covers(df)]


def get_neigborhood(graph, feat, nodes, radius):
    dense_adj = to_dense_adj(torch.tensor(graph)).numpy()[0]

    g = nx.from_numpy_matrix(dense_adj)  # nx.Graph()
    g.add_nodes_from([(i, {"x": el}) for i, el in enumerate(np.argmax(feat, axis=1))])
    # g.add_edges_from(graph.transpose().tolist())

    mol = g  # nx.from_numpy_matrix(dense_adj)

    nn = nodes.copy()
    for i in range(radius):
        nn += [el for n in nn for el in list(mol.adj[n])]
        nn = list(set(nn))
    nbh = nx.Graph(mol.subgraph(nn).copy())
    for n in nodes:
        nbh.nodes[n]["x"] += 100
    return nbh


def get_egos(datas, activations, motif):
    ids = get_covers(activations, motif)
    act_ids = list()
    for g_id, (graph, features, label) in enumerate(tqdm(datas)):
        if len(ids.index[ids["id"] == g_id]) == 0:
            continue
        i0 = activations.index[activations["id"] == g_id][0]
        indexes = ids.index[ids["id"] == g_id].tolist()
        for i in indexes:
            act_ids.append((get_neigborhood(graph, features, [i - i0], int(motif[1][2]) + 1), 1))

    return act_ids
    # TODO


def get_negative(datas, activations, motif):
    ids = get_covers(activations, motif)
    act_ids = list()
    for g_id, (graph, features, label) in enumerate(tqdm(datas)):
        if len(ids.index[ids["id"] == g_id]) == 0:
            dense_adj = to_dense_adj(torch.tensor(graph)).numpy()[0]

            g = nx.from_numpy_matrix(dense_adj)  # nx.Graph()
            g.add_nodes_from([(i, {"x": el}) for i, el in enumerate(np.argmax(features, axis=1))])
            # g.add_edges_from(graph.transpose().tolist())

            act_ids.append((g, 0))

    return act_ids


def save_egos(egos, dataset, motif_id):
    with open("results/egos/" + dataset + "_" + str(motif_id) + "labels_egos.txt", "w") as f:
        for i, (g, l) in enumerate(egos):
            f.write("t # " + str(i) + "# " + str(l) + "\n")
            for n in g.nodes:
                label = g.nodes[n]["x"]
                f.write("v " + str(n) + " " + str(label) + "\n")
            for a in g.adj:
                for b in g.adj[a]:
                    f.write("e " + str(a) + " " + str(b) + " 0 \n")
        f.write("t # -1")


def get_dataset_prior(dataset, mol_level=False):
    graphs, features, labels, _, _, _ = load_dataset(dataset)
    return np.concatenate(features, axis=0).sum(axis=0) / features.sum()
    #


def to_gspan(dataset, motif_id):
    graphs, features, labels, _, _, _ = load_dataset(dataset)
    datas = list(zip(graphs, features, labels))
    activation = load_activations(dataset)
    motifs = load_motifs(dataset)[motif_id]
    egos = get_egos(datas, activation, motifs)
    negative = get_negative(datas, activation, motifs)
    save_egos(egos + negative, dataset, motif_id)


from subgraph_metrics import get_metric


def gspan_metrics(dataset):
    metric = exact_ged
    out = compare_motifs(dataset, metric)
    return out


def self_metric(dataset):
    metric = common_subgraph
    out = get_metric(dataset, metric)
    return out


##MAIN
from ExplanationEvaluation.explainers.XGNN_explainer import gnn_explain
from ExplanationEvaluation.models.model_selector import model_selector
from ExplanationEvaluation.explainers.MCTS_explainer import MCTSExplainer
from ExplanationEvaluation.gspan_mine.gspan_mining.gspan import GSpanMiner
from ExplanationEvaluation.explainers.VAE import GVAE
from ExplanationEvaluation.explainers import VAE
from ExplanationEvaluation.gspan_mine.gspan_mining.gspan import GSpanMiner
from ExplanationEvaluation.explainers.RandomExplainers import RandomExplainer_subgraph, get_activation_score



Number_of_rules = dict([("aids", 60), ('mutag', 60), ("BBBP", 60), ("PROTEINS_full", 28), ("DD", 47), ("ba2", 19)])


def run_gspan(dataset):
    model, checkpoint = model_selector("GNN",
                                       dataset,
                                       pretrained=True,
                                       return_checkpoint=True)
    results = {}
    explainer = GSpanMiner(dataset, model, 0,
                           use_up2=True
                           )
    #with open(f'results/dataframes/{dataset}_gspan.csv', 'w+') as f:
        #dictwriter_object = DictWriter(f, fieldnames=["dataset", "algorithm", "graph", "metric", "episode", "rule", "value"])

        #dictwriter_object.writeheader()
    log = pd.DataFrame(columns=["dataset", "algoritm", "metric", "episode", "rule", "value"])
    rules = [0]
    rules = range(0,Number_of_rules[dataset])
    for target_rule in tqdm(rules):
        if explainer.rules[target_rule]!= explainer.target_rule:
            explainer.change_target_rule_and_reset(target_rule)
        explainer.run()
        result = explainer.report_best_one()
        result["rule"] = target_rule
        """"for k, v in result["scores"].items():
            res = {"dataset": dataset,
                       "algoritm": "gspan",
                      "metric" :k ,
                      "episode": 0,
                      "rule" :target_rule,
                      "value": v
            }
            #res = [pd.Series([dataset,"gspan", k, 0, target_rule, v], index=log.columns)]
            #log=log.append(res)"""
        with open("results/gspan/gspan_" + dataset + "_rule_" + str(target_rule)+".pkl" ,"wb") as f:
            pickle.dump(result, f)
        #dictwriter_object.writerow(result)
        # xx = explainer.tree.sort_nodes(explainer=explainer)
        # xx = explainer.tree.sort_nodes(explainer=explainer)[10]
        # roll = explainer.best_rollouts(10)

        #f.close()
        #log.to_csv("results/dataframes/gspan_" + dataset + ".csv")

    #result = pd.read_csv(f"results/dataframes/{dataset}_gspan.csv")

    return result


'''    col_names = list(results[0].keys())
    dic = {i : [results[i][key] for key in col_names] for i in range(60)}
    df = pd.DataFrame.from_dict(dic, columns=col_names, orient='index')
    df.to_csv(f"{dataset}")
    with open(f'result{dataset}.json', 'w') as fp:
        json.dump(results, fp)'''

from collections import defaultdict


def run_random(dataset):

    graphs, features, labels, _, _, _ = load_dataset(dataset)
    model, checkpoint = model_selector("GNN",
                                       dataset,
                                       pretrained=True,
                                       return_checkpoint=True)
    metrics = ["sum", "lin", "entropy", "likelyhood_max"]

    for rule in range(Number_of_rules[dataset]):
        res_metrs = defaultdict(lambda: defaultdict(list))
        log = pd.DataFrame(columns=["dataset", "algoritm", "metric", "episode", "rule", "value"])
        for metric in metrics:
            explainer = RandomExplainer_subgraph(model, (graphs, features, labels), dataset, target_rule=rule )
            explainer.prepare()

import pickle

from functools import partial

def rule_value(evaluator,metr, rule, node):
    rulenb =  evaluator.rules[rule]
    rule = evaluator.funcs[metr](node.emb, rulenb)
    return rule

def get_children_values(tree, fun):
    nodes = tree.as_list()
    nodes = [el for el in nodes if el.emb is not None]

    vals = list()
    for node in nodes:
        if node.emb is not None:
            v1 = fun(node)
            for el in node.children:
                if el.emb is not None:
                    vals.append(np.abs(v1-fun(el)))
    return np.mean(vals)

def get_distribution_values(tree,evaluator, metr, rules, rule):
    nodes = tree.as_list()
    #nodes = sorted([el for el in nodes if el.emb is not None])[-5:]
    nodes = sorted([el for el in nodes if el.emb is not None], key = (lambda el: evaluator.funcs[metr](el.emb, evaluator.rules[rule])))[-10:]

    vals =list()

    for r in rules:
        rulenb = evaluator.rules[r]
        rr= list()
        for node in nodes:
            vv = evaluator.funcs[metr](node.emb, rulenb)
            rr.append(vv)
        vals.append((np.mean(rr),np.std(rr)))
    return vals



def run_mcts(dataset):
    graphs, features, labels, _, _, _ = load_dataset(dataset)

    if dataset =="ba2":
        edge_probs=None
    else :
        if dataset =="PROTEINS_full":
            edge_probs = get_edge_distribution(graphs, features,labels,30)
        else :
            edge_probs = get_edge_distribution(graphs, features,labels)
        #degre_distrib = get_degre_distribution(graphs, features)

    model, checkpoint = model_selector("GNN",
                                       dataset,
                                       pretrained=True,
                                       return_checkpoint=True)
    metrics = ["cosine","entropy", "likelyhood_max"]
    #metrics = ["likelyhood_max"]

    real_ratio = {"cosine": (0, 1),
                  "entropy" : (-70,-2.014),
                  "lin" : (-1,0.1),
                  "likelyhood_max": (-35,20.04)
                  }
    #metrics = ["likelyhood_max"]
    eval_metrics = []#metrics[0]
    rules = [range(Number_of_rules[dataset])]



    scores = list()
    nsteps = 4000
    nxp = 4
    #print( dataset+ " "+ str(nsteps) + " " + str(nxp)+ " " + str(metrics))
    R=[0]
    r=0
    #ratios = [coef/5 for coef in range(5)]+[1]
    ratios = [0.5]
    print("run MCTS dataset :" + str(dataset)+ " nxp :" +str(nxp) + " budget : "+ str(nsteps))

    for r in ratios:# [1 / (2 ** coef) for coef in range(10)]+[0]:
        for rule in tqdm(rules[0]):
            #res_metrs = defaultdict(lambda: defaultdict(list))
            #log = pd.DataFrame(columns=["dataset", "algoritm", "metric", "episode", "rule", "value"])

            for metric in metrics:
                for x in (range(nxp)):
                    explainer = MCTSExplainer(model, (graphs, features, labels), 6, 10, 1, dataset, target_rule=rule,
                                                  target_metric=metric, uid=x, edge_probs=edge_probs,
                                                  real_ratio= (real_ratio[metric],r))
                    explainer.train(nsteps)
                    with open("results/mcts_dumps_new/dataset_" + dataset +"_rule_" + str(rule)+"metric_"+metric+"xp_" +str(x)+"steps_"+str(nsteps)+ "nxp_"+ str(nxp)+"ratio_"+str(r)+".pkl", 'wb') as f1:
                        explainer.tree.root.clean_tree()
                        nl = [el for el in explainer.tree.root.as_list() if el.own_value_detailed is not None]

                        """nodes = list(sorted(nl, key=lambda x: x.own_value))[-10:]

                        for n in nodes:
                            n.parent=None
                            n.children=None

                        #pickle.dump(nodes, f1)"""

                        pickle.dump(explainer.tree.root, f1)

    print(0)

def get_pareto(nodes):
    pareto = np.ones(len(nodes))
    for i, el in enumerate(nodes):
        if  pareto[i]:
            for n in np.where(pareto)[0]:
                if n>i:
                    if nodes[n][1]>el[1] and nodes[n][2]> el[2]:
                        pareto[i] = 0
                        break
                    if nodes[n][1] <=el[1] and  nodes[n][2]< el[2]:
                        pareto[n] = 0
    return [nodes[i] for i in  np.where(pareto)[0]]

def run_vae(dataset):
    graphs, features, labels, _, _, _ = load_dataset(dataset)
    model, checkpoint = model_selector("GNN",
                                       dataset,
                                       pretrained=True,
                                       return_checkpoint=True)
    vae = GVAE(k=38, feature_size=14, dataset_name=dataset, target_rule=0, model=model, adjs=graphs, features=features)
    VAE.train(vae)


def run_xgnn(dataset, random = False):
    graphs, features, labels, _, _, _ = load_dataset(dataset)
    model, checkpoint = model_selector("GNN",
                                       dataset,
                                       pretrained=True,
                                       return_checkpoint=True)

    metrics = [ "cosine", "entropy", "likelyhood_max"]
    budget = 5000
    nxp = 2
    rules = range(Number_of_rules[dataset])
    #rules = [28]

    if dataset =="ba2":
        edge_probs=None
    else :
        if dataset =="PROTEINS_full":
            edge_probs = get_edge_distribution(graphs, features,labels,30)
        else :
            edge_probs = get_edge_distribution(graphs, features,labels)
        #degre_distrib = get_degre_distribution(graphs, features)


    real_ratio = {"cosine": (0, 1),
                  "entropy" : (-70,-2.014),
                  "lin" : (-1,0.1),
                  "likelyhood_max": (-35,20.04)
                  }
    ratios = [1 / (2 ** coef) for coef in range(10)]+[0,0.75]
    ratios = [coef/5 for coef in range(5)]+[1]
    if random:
        ratios = [0.5]
    #ratios = [0.5]
    print("run XGNN dataset :" + str(dataset)+  " nxp :" +str(nxp) + " budget : "+ str(budget) +" random" + str(random))

    for ratio in ratios:
        for rule in tqdm(rules):
            for metric in metrics:
                #print("Random : ", r)
                for i in range(nxp):
                    out_graphs = list()

                    explainer = gnn_explain(model, (graphs, features, labels), 6, 10, 1, dataset, random=random,
                                            target_rule=rule,  target_metric=metric,
                                            real_ratio=(real_ratio[metric],ratio), edge_probs=edge_probs)
                    out_graphs.append(explainer.train(budget))
                    #explainer.graph_draw(explainer.graph)
                    if random :
                        file= "results/random_dumps_new/dataset_" + dataset + "_rule_" + str(
                            rule) + "metric_" + metric + "xp_" + str(i) + "steps_" + str(budget) + "nxp_" + str(nxp)+"ratio_"+str(ratio)+"random.pkl"
                    else :
                        file = "results/xgnn_dumps_new/dataset_" + dataset + "_rule_" + str(
                            rule) + "metric_" + metric + "xp_" + str(i) + "steps_" + str(budget) + "nxp_" + str(nxp)+"ratio_"+str(ratio)+".pkl"

                    with open(file, 'wb') as f1:
                        pickle.dump(out_graphs,f1)




def get_dataset_distibution(dataset):
    graphs, features, labels, _, _, _ = load_dataset(dataset)
    model, checkpoint = model_selector("GNN",
                                       dataset,
                                       pretrained=True,
                                       return_checkpoint=True)

    metrics = ["sum", "lin", "entropy", "cheb", "likelyhood", "likelyhood_max", "hamming"]
    log = pd.DataFrame(columns=["dataset", "algoritm", "metric", "episode", "rule", "value"])

    for rule in tqdm(range(Number_of_rules[dataset])):
        #log = pd.DataFrame(columns=["dataset", "algoritm", "metric", "episode", "rule", "value"])
        act, unact = get_activation_score(model, (graphs, features, labels), dataset,rule, metrics)
        vals = list()
        d = {"act": act, "unact":unact}
        for name, act in d.items():
            for el in act:
                for k, v in el.items():
                    if k in metrics:
                        vals.append(pd.Series([dataset, "rule_"+name, k, -1, rule, v], index=log.columns))
        log = log.append(vals)
        log.to_csv("results/dataframes/dataset_distibution_2_" + dataset + ".csv")

        """
                explainer.train(1)
                # explainer.graph_draw(explainer.graph)

                for k, v in explainer.log.items():
                    if k in metrics:
                        vals = [pd.Series([dataset, "random_", k, i, rule, x], index=log.columns)
                                for i, x in enumerate(v)]
                        log = log.append(vals)
                    res_metrs[k][metric].append(v)

                # plt.show()
                # print(explainer.losses)
        log.to_csv("results/dataframes/dataset_distibution_" + str(rule) + "_" + dataset + "csv")"""
"""
for dataset in ["aids" ,"mutag","BBBP", "DD", "PROTEINS_full"]:
    run_xgnn(dataset)
    """#get_dataset_distibution(dataset)



if __name__ == '__main__':
    #for dataset in ["aids","mutag","BBBP", "DD", "PROTEINS_full","ba2"]:

    #for dataset in []:

    for dataset in ["aids"]:
        #run_vae(dataset)

        #get_dataset_distibution(dataset)
        #run_random(dataset)
        #run_xgnn(dataset, random=False)
        #run_mcts(dataset)

        run_gspan(dataset)


def inside_explain():
    exceptions = {("ba2", 5), ("ba2", 15)}

    Datasets = [("aids", 60), ('mutag', 60), ("BBBP", 60), ("PROTEINS_full", 29), ("DD", 60), ("ba2", 19)]

    for (dset, R) in Datasets:
        for r in range(R):
            results = list()
            to_gspan(dset, r)
            if (dset, r) not in exceptions:
                results.append(run_dset(dset, r))
        with open("results/subgraphs/" + dataset + "_results.json", 'w+') as json_file:
            json_file.write(json.dumps(results))

        if dset == "mutag" and True:
            print('comparasion between mutag and XGNN')
            print(gspan_metric(dset))
        print("common subgraph between patterns and dataset")
        print(self_metrics(dset, r))






