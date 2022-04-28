import torch
import pandas as pd
from tqdm import tqdm
import pysubgroup as ps
from torch_geometric.utils import to_dense_adj, dense_to_sparse
import numpy as np
import networkx as nx
from os.path import isfile, join
import os
from os import listdir
import itertools
#import seaborn as sns
import copy
import matplotlib.patches as mpatches

from ExplanationEvaluation.explainers.utils import RuleEvaluator, get_atoms, get_edge_distribution
from ExplanationEvaluation.explainers.MCTS_explainer import Node
from ExplanationEvaluation.datasets.dataset_loaders import load_dataset

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import figure
from scipy.stats import norm
from scipy.stats import gaussian_kde

from ExplanationEvaluation.models.model_selector import model_selector
from subgraph_metrics import *
from itertools import  product


import random
import pickle
Number_of_rules = dict([("aids", 60), ('mutag', 60), ("BBBP", 60), ("PROTEINS_full", 28), ("DD", 47), ("ba2", 19)])
in_dir = "mcts_dumps_new"
out_dir = "learning_curve_new"

epochs = 4000

#datasets = ["aids", "mutag",  "BBBP","DD","PROTEINS_full"]

datasets = ["aids", "mutag",  "BBBP","ba2", "DD" , "PROTEINS_full"]
datasets =["ba2"]
nxp = 4

dnames = ["aids", "mutag",  "BBBP","ba2" ,"DD", "Proteins"]
#ratio = 0.5


real_ratio = {"cosine": (0, 1),
              "entropy": (-70, -2.014),
              "lin": (-1, 0.1),
              "likelyhood_max": (-35, 20.04)
              }


def get_tree(dataset, rule, metric= "lin", steps=None, nxp=None, nbxp=None, ratio=None):
    out= list()
    if nxp is None:
        nxp = 10
    if nbxp is None:
        nbxp=nxp
    for xp in range(nbxp):
        if steps is not None:
            file = "results/"+ in_dir +"/dataset_" + dataset +"_rule_" + str(rule)+"metric_"+metric+ "xp_" +str(xp)+"steps_"+str(steps)+ "nxp_"+ str(nxp)+".pkl"
        else:
            file = "results/"+ in_dir +"/dataset_" + dataset +"_rule_" + str(rule)+"metric_"+metric+ "xp_" +str(xp)+".pkl"
        if ratio is not None:
            file = "results/"+ in_dir +"/dataset_" + dataset +"_rule_" + str(rule)+"metric_"+metric+ "xp_" +str(xp)+"steps_"+str(steps)+ "nxp_"+ str(nxp)+"ratio_"+str(ratio)+".pkl"

        ff= getfile(file)
        out.append(ff)
    return out


def getfile(filename):
    with open(filename, "rb") as f1:
        val= pickle.load(f1)
    return val
def sort_nodes(nodes, value_function):
    node_list = nodes.as_list()
    return sorted([(node, value_function(node)) for node in node_list if node.visit and node.emb is not None], key= lambda x:x[1])


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


def extract_best_patterns():
    print("extract_pattens")

    metrics = ["cosine", "entropy", "likelyhood_max"]
    # metrics = ["likelyhood_max"]
    nxp = 6
    cols = ["dataset", "rule", "xp", "step", "layer", "metr_name", "metr_val", "metr_score", "metr_real"] + [
        "c_" + str(i) for i in range(20)]

    for dataset in datasets:
        graphs, features, labels, _, _, _ = load_dataset(dataset)
        datas = (graphs, features, labels)

        #rules = range(0, Number_of_rules[dataset], 5)
        rules = range(Number_of_rules[dataset])
        #rules = [10, 12, 14, 15, 16, 18]

        df = list()
        for rule in tqdm(rules):

            for metric in metrics:
                if small:
                    for xp in range(nxp):

                        file = "results/" + in_dir + "_small/dataset_" + dataset + "_rule_" + str(
                            rule) + "metric_" + metric + "xp_" + str(xp) + "steps_" + str(epochs) + "nxp_" + str(
                            nxp) + "ratio_" + str(ratio) + ".pkl"
                        with open(file, "rb") as f1:
                            val = pickle.load(f1)
                        best = val[-1]
                        scores = []
                        plt()


def to_dataframe(small=False):
    print("to_dataframe")

    metrics = ["cosine", "entropy","likelyhood_max"]
    #metrics = ["likelyhood_max"]
    nxp = 4
    cols = ["dataset", "rule", "xp", "step", "layer", "metr_name", "metr_val","metr_score", "metr_real"] + ["c_" + str(i) for i in range(20)]

    for dataset in datasets:
        graphs, features, labels, _, _, _ = load_dataset(dataset)
        datas = (graphs, features, labels)

        #rules = range(0, Number_of_rules[dataset], 5)
        rules = range(Number_of_rules[dataset])
        #rules = [10, 12, 14, 15, 16, 18]

        df = list()
        for rule in tqdm(rules):

            for metric in metrics:
                if small:
                    for xp in range(nxp):

                        file = "results/" + in_dir + "_small/dataset_" + dataset + "_rule_" + str(
                            rule) + "metric_" + metric + "xp_" + str(xp) + "steps_" + str(epochs) + "nxp_" + str(
                            nxp) + "ratio_" + str(ratio) + ".pkl"
                        with open(file, "rb") as f1:
                            val = pickle.load(f1)
                        best = val[-1]
                        scores = []

                        df.append([dataset, rule, xp, best.step, best.layer, metric, best.own_value, best.own_value_detailed[0],
                                   best.own_value_detailed[1]] + scores + best.emb.tolist())
                else:
                    tree = get_tree(dataset, rule,metric= metric, steps= epochs, nxp=nxp, ratio=0.5)
                    for i, t in enumerate(tree):
                        #t.clean_tree()
                        nl = t.as_list()
                        #max_epoch = max([el.step for el in nl if el.emb is not None])
                        for el in nl:
                            if el.own_value is not None:
                                scores = []
                                df.append([dataset, rule, i, el.step, el.layer, metric,el.own_value,el.own_value_detailed[0],el.own_value_detailed[1]]+ scores + el.emb.tolist())

        df = pd.DataFrame(df,
            columns=cols)
        if small :
            df.to_csv("results/"+out_dir+"/small_dataframes/"+ dataset+".csv")
        else :
            df.to_csv("results/" + out_dir + "/dataframes/" + dataset + ".csv")





def latex_metrics(data, error, name, std=True):

    l = str(name)
    for v, er in zip(data, error):
        if v < 10000:
            vv = format(v ,'.2f')
            rr = format(er,'.2f')
        else:
            vv = format(v ,'.2e')
            rr = format(er,'.2e')
        if std:
            l += " & $ " + vv +"\pm"+rr +"$"
        else:
            l += " & $ " + vv + "$"
    l+="\\\\\\hline"
    print(l)

def get_best_node(node, value_function):
    max = value_function(node)
    best_node = node
    for i, ch in enumerate(node.children):
        n, v = get_best_node(node, value_function)
        if activate(node.emb) and v> max:
            max = v
            best_node = ch
    return best_node, max

def sort_by_epoch(x):
    return x.step


def experiment_learnig_curve_to_df():

    metrics = ["cosine", "entropy","likelyhood_max"]
    #metrics = ["cosine", "lin"]

    convolve = 50
    confidence=0.95
    tval = norm.ppf((1+confidence)/2)
    #rules = range(Number_of_rules[dataset])
    rules = range(0,Number_of_rules[dataset], 5)

    for metric in metrics:
        for i, dataset in enumerate(datasets):
            values = list()

            for rule in tqdm(rules):
                evaluator = RuleEvaluator(None, dataset,datas=datas, target_rule=rule,metrics=metric)

                trees = get_tree(dataset,rule)

                node_list=[ sort_nodes(t, sort_by_epoch) for t in trees]

                values += [[evaluator.compute_score_emb(n.emb, metric) for n, _ in nl] for nl in node_list]#passer à metric
            vals = np.array(values)
            with open("results/" +out_dir+"/"+metric+"_"+dataset+".pkl","wb") as f:
                pickle.dump(vals, f)


from matplotlib.pyplot import figure
import operator
from functools import reduce
def experiment_learnig_curve_plot():
    # rules = range(4)
    #dataset", "rule", "xp", "step", "metr_name", "metr_val","metr_score", "metr_real
    #datasets = ["aids"]
    metrics = ["cosine", "entropy","likelyhood_max"]#, "lin" , "likelyhood_max"]

    mnames = ["Cosine", "Cross-Entropy", "Relative-CE"]# ,"Relative-CE"]
    dnames = {"aids": "aids", "mutag": "Mutagen",  "BBBP": "BBBP","ba2":"BA2" ,"DD":"DD", "PROTEINS_full":"Proteins"}

    metric_names = dict(zip(metrics, mnames))
    dset_names = dict(zip(datasets, dnames))

    convolve = 50
    n_epoch = 400
    confidence=0.98
    tval = norm.ppf((1+confidence)/2)

    target_value = ["metr_val"]

    for metric in metrics:

        if True:
            #for layer in range(3):
            plt.figure(figsize=(15, 10),)
            for i, dataset in enumerate(datasets):
                rulenb = [i for i, e  in  enumerate(RuleEvaluator.load_rules(dataset)) if e[0]>=1]
                f = "results/" + out_dir + "/dataframes/" + dataset + ".csv"
                df = pd.read_csv(f)

                df = df[reduce(operator.or_, [df["rule"] == r for r in rulenb], False)]

                for t in target_value:
                    # rules = range(Number_of_rules[dataset])
                    #rules = [10, 12, 14, 15, 16, 18]
                    rules = range(Number_of_rules[dataset])

                    #for r in rules:

                    ddf = df[(df["metr_name"] == metric)]
                    dfi = list()
                    for r in rules:
                        dfi.append(ddf[ddf["rule"]==r])
                    dfc = pd.concat(dfi)
                    #vals = df[(df["metr_name"] == metric) ].groupby("step")["metr_val"]
                    #metr_val","metr_score", "metr_real
                    vals= dfc.groupby("step")[t]
                    vv = dfc.groupby(["rule", "xp"])
                    nels = np.array([len(v[1]) for v in vals ])

                    vv= [dfc.loc[v] for v in vv.groups.values()]
                    #v= [max([vv for v in v[—] for v in vv]

                    values = np.zeros((len(vv),epochs))
                    for j, v in enumerate(vv):
                        maxv = -1
                        for i in range(1,max(v["step"])) :
                            #print(i,v["step"][i])
                            cur = v.loc[v["step"].index[i]][t]
                            if cur>maxv:
                                maxv= cur
                            values[j,i]=maxv
                    # v= [[max(v[v["step"] <= i][t]) for i in range(1,max(vv[0]["step"]))] for v in vv]
                    #values = vals.mean().tolist()
                    values = values[:, :-2]
                    v = values.mean(axis=0)
                    std = values.std(axis=0)/np.sqrt(values.shape[0])

                    #v = np.convolve(values,np.ones(convolve), "valid")/convolve

                    #std = np.convolve(np.array(std)*tval, np.ones(convolve), "valid")/convolve

                    plt.errorbar(range(len(v)), v, std, label=dnames[dataset],elinewidth=1, errorevery=(i*50, 100))

            #plt.xlabel("Epoch",fontsize=15)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)

            plt.legend(loc='lower right',fontsize=20)

            #plt.title(dataset)


            plt.savefig("results/figs/learning_croissant" + metric + ".png")

            plt.show()




def test_similarities():
    metrics = ["cosine", "entropy", "likelyhood_max"]
    # metrics = ["likelyhood_max"]
    nxp = 2
    cols = ["dataset", "rule", "xp", "step", "layer", "metr_name", "metr_val", "metr_score", "metr_real"] + [
        "c_" + str(i) for i in range(20)]

    for dataset in datasets:
        graphs, features, labels, _, _, _ = load_dataset(dataset)
        datas = (graphs, features, labels)

        # rules = range(0, Number_of_rules[dataset], 5)
        rules = range(Number_of_rules[dataset])
        # rules = [10, 12, 14, 15, 16, 18]

        df = list()
        bests_graphs =list()
        for rule in tqdm(rules):
            #evaluator = RuleEvaluator(None, dataset, datas=datas, target_rule=rule, metrics=metrics)
            bests_graphs.append({m: list()for m in metrics})
            for metric in metrics:
                tree = get_tree(dataset, rule, metric=metric, steps=epochs, nxp=nxp, nbxp=2)
                for i, t in enumerate(tree):
                    # t.clean_tree()
                    nl = [el for el in t.as_list() if el.own_value_detailed is not None]
                    for i in range(len(nl)):
                        nl[i].children=None
                        nl[i].parent=None
                    best=sorted(nl, key=lambda x: x.own_value)
                    bests_graphs[rule][metric].append(best[-1])
                    # max_epoch = max([el.step for el in nl if el.emb is not None])
        print("bbb")

def get_node_list(graph):
    return [graph.nodes[el]["label"] for el in graph.nodes]

def get_dset_embeddings(dataset):
    graphs, features, labels, _, _, _ = load_dataset(dataset)
    model,_ = model_selector("GNN", dataset, pretrained=True, return_checkpoint=True)


    emb= [model.embeddings(torch.tensor(feat),torch.tensor(adj))
                                         for adj, feat in zip(graphs, features)]
    #datas = [dset_to_nx(g, x).subgraph(list(range(np.argmin(x.sum(axis=1))))) for g, x in zip(graphs, features)]
    emb = [[e.detach() for g, f in zip(emb, features) for e in g[i][:int(f.sum())]] for i in range(3)]
    return emb, model, (graphs, features, labels)


def get_dset_egos(dataset):
    graphs, features, labels, _, _, _ = load_dataset(dataset)
    model,_ = model_selector("GNN", dataset, pretrained=True, return_checkpoint=True)


    grs = [dset_to_nx(g,x, atom_map=get_atoms(dataset,(graphs, features) )).subgraph(list(range(np.argmin(x.sum(axis=1))))) for g,x in zip(graphs,features)]
    emb= [model.embeddings(torch.tensor(feat),torch.tensor(adj))
                                         for adj, feat in zip(graphs, features)]
    egos = [[e for g in tqdm(grs) for e in get_ego(g, l+1)] for l in range(3)]

    #datas = [dset_to_nx(g, x).subgraph(list(range(np.argmin(x.sum(axis=1))))) for g, x in zip(graphs, features)]
    emb = [[e.detach() for g, f in zip(emb, features) for e in g[i][:int(f.sum())]] for i in range(3)]
    return emb,egos,  model, (graphs, features, labels)
def get_ego(graph,radius):

    dists = dict(nx.all_pairs_shortest_path_length(graph))
    d2 = np.ones((len(graph),len(graph)))*100
    for i, el in dists.items():
        for j, v in el.items():
            d2[i,j] = v
    out = list()
    for node in graph:
        fil =  filter(lambda x :d2[node,x]<=radius, graph )
        ego= nx.Graph(graph.subgraph(fil))
        for u,v in ego.edges:
            if d2[node, u]>= radius and d2[node, v]>= radius:
                ego.remove_edge(u,v)
        out.append((ego,node))
    return out




def experiment_2_quartiles():

    number_of_quantiles = 4

    metrics = ["cosine", "entropy","likelyhood_max"]
    mnames = ["Cosine", "Cross-Entropy", "Relative-CE"]# ,"Relative-CE"]

    metric_names = dict(zip(metrics, mnames))
    dset_names = dict(zip(datasets, dnames))

    confidence=0.95
    tval = norm.ppf((1+confidence)/2)

    for dataset in datasets :
        embs, model,datas = get_dset_embeddings(dataset)
        f = "results/"+out_dir+"/dataframes/" + dataset + ".csv"
        df = pd.read_csv(f)

        for metric in metrics:
            out_support = list()
            out_random = list()
            rules = range(0, Number_of_rules[dataset])

            for rule in rules:
                eval = RuleEvaluator(model, dataset, datas, rule, [metric])

                support = list(filter(eval.activate,embs[eval.get_layer()]))
                support = random.sample(support, min(100,len(support)))
                support = np.array([eval.compute_score_emb(emb,metric) for emb in support])

                rnd = random.sample(embs[eval.get_layer()], len(support))
                rnd = np.array([eval.compute_score_emb(emb, metric) for emb in rnd])


                minv = min(support.min(),rnd.min())-1e-10
                maxv = min(support.max(),rnd.max())

                if False:#metric in ["entropy","likelyhood_max"]:
                    support -=1
                    rnd -= 1
                else:
                    support = (support - minv) / (maxv - minv)
                    rnd = (rnd-minv)/(maxv- minv)


                support_quantiles =[ np.quantile(support,k/number_of_quantiles) for k in range(number_of_quantiles)]
                random_quantiles = [ np.quantile(rnd,k/number_of_quantiles) for k in range(number_of_quantiles)]

                model_vals = df[(df["metr_name"]==metric)& (df["rule"]==rule)].groupby("xp")["metr_val"].max().tolist()
                #model_vals = [el[1] for el in node_list]

                if False: #metric in ["entropy","likelyhood_max"]:
                    rescale = lambda x: x-1

                else:
                    rescale = lambda x: (x - minv) / (maxv - minv)

                values_v_support = [[(rescale(v)/q) for q in support_quantiles] for v in model_vals ]
                values_v_random = [[(rescale(v)/q) for q in random_quantiles]for v in model_vals ]

                out_support += (values_v_support)
                out_random += (values_v_random)

            support = np.array(out_support).mean(axis=0)
            support_std = np.array(out_support).std(axis=0)

            rnd = np.array(out_random).mean(axis=0)
            rnd_std = np.array(out_random).std(axis=0)
            latex_metrics(support,support_std , dset_names[dataset] +" " + metric_names[metric] + "& support ",std=False)
            latex_metrics(rnd,rnd_std ,dset_names[dataset] +" " + metric_names[metric] + " & rnd " , std=False)
        print("")

def experiment_3_l2():
    print("experiment 3 l2")

    metrics = ["cosine", "entropy","likelyhood_max"]#["cosine", "entropy", "lin","likelyhood_max"]


    convolve = 50
    confidence=0.95
    tval = norm.ppf((1+confidence)/2)

    dis = "min"
    tot_support = defaultdict(list)
    tot_rnd = defaultdict(list)
    fig, axs = plt.subplots( len(datasets), len(metrics),  figsize=(9, 18), sharex=True, sharey=True,
                            tight_layout=True)
    for dset_i, dataset in enumerate(datasets) :

        embs, model,datas = get_dset_embeddings(dataset)

        f = "results/"+out_dir+"/dataframes/" + dataset + ".csv"
        df = pd.read_csv(f)

        for metr_i, metric in enumerate(metrics):
            out_support = list()
            out_random = list()
            rules = range(0, Number_of_rules[dataset])

            for rule in rules:
                eval = RuleEvaluator(model, dataset, datas, rule, [metric])

                #support = embs[eval.get_layer()]
                support = list(filter(eval.activate,embs[eval.get_layer()]))
                support = random.sample(support, min(100,len(support)))

                rnd = random.sample(embs[eval.get_layer()], len(support))
                #rnd = [eval.compute_score_emb(emb, metric)[metric]for emb in rnd]

                els= df[(df["metr_name"] == metric) & (df["rule"] == rule)].groupby("xp")
                values= [np.array([df.iloc[i]["c_"+str(j)] for j in range(20)]) for el, i in zip(els, els["metr_val"].idxmax())]

                support = np.array([el.numpy()for el in support])

                rnd = np.array([el.numpy()for el in rnd])

                d_support = [((support - v)**2).sum(axis=1) for v in values]
                d_rnd= [((rnd - v)**2).sum(axis=1) for v in values]

                """if dis =="min":
                    d_support = np.min(d_support)
                    d_rnd = np.mean(d_rnd)
                else:
                    d_support = d_support.mean()
                    d_rnd = d_rnd.mean()
                """
                out_support.append((np.min(d_support),np.mean(d_support)))
                #out_support
                out_random.append((np.min(d_rnd),np.mean(d_rnd)))


            tot_support[metric] += out_support
            tot_rnd[metric] += out_random
            support = np.array(out_support).mean(axis=0)
            support_std = np.array(out_support).std(axis=0)

            rnd = np.array(out_random).mean(axis=0)
            rdn_std = np.array(out_random).std(axis=0)


            #print(dataset +" " + metric)
            #print(support)
            #print(support_std)

            #print(rdn)
            #print(rdn_std)
            color= ["r","c"]#,"r","c"]
            toplot = [support[1], rnd[1]]
            toplot_std = [support_std[1], rdn_std[1]]

            axs[dset_i, metr_i].bar([0, 0.5],height= toplot,color = color, yerr= toplot_std, width = 0.4,capsize = 5)
            axs[dset_i, metr_i].set_xticklabels([])
            #axs[dset_i,metr_i].bar(0.25,height= rnd[0],color = 'r', yerr= rdn_std[0],width = 0.1,capsize = 10)

            #plt.bar([0,0.25, 0.5, 0.75],height= toplot,color = color, yerr= toplot_std, width = 0.15,capsize = 10)

            #plt.show()
            #axs[dset_i,metr_i].bar(x=["support", "random"],height= support+rnd, yerr= support_std+rdn_std)

            #axs[dset_i,metr_i].set_title(dataset+ " "+metric)
            latex_metrics(support+rnd, support_std+rdn_std ,"xp3 " +dataset +" " + metric + " ")
            #latex_metrics(rnd,rdn_std ,"xp3 " +dataset +" " + metric + " rnd " )


    red_patch = mpatches.Patch(color='r', label='support')
    cian_patch = mpatches.Patch(color='c', label='random')

    plt.legend(handles=[red_patch,cian_patch])
    #plt.legend(labels=[ 'support min', 'random mean'])
    plt.show()


    for k in tot_support.keys():
        support = np.array(tot_support[k]).mean(axis=0)
        support_std = np.array(tot_support[k]).std(axis=0)

        rnd = np.array(tot_rnd[k]).mean(axis=0)
        rdn_std = np.array(tot_rnd[k]).std(axis=0)

        latex_metrics(support+rnd, support_std+rdn_std, "xp3 " + dataset + " " + k + " ")
        #latex_metrics(rnd, rdn_std, "xp3 " + dataset + " " + k + " rnd ")
    print("experiment 3 l2 over ")


def experiment_3_bis():
    print("3 bis GED")


    metrics=["cosine", "entropy", "likelyhood_max"]

    confidence=0.95
    tval = norm.ppf((1+confidence)/2)
    #rules = range(0,Number_of_rules[dataset], 5)
    rule_interval = 1
    graphs = list()#[list() for _ in metrics]

    for dataset in datasets :
        graphs.append([list() for _ in range(0,Number_of_rules[dataset],rule_interval)])

        embs, egos, model, (gr, features, labels) = get_dset_egos(dataset)

        if dataset =="PROTEINS_full":
            edge_probs = get_edge_distribution(gr, features,labels, 30)
        else :
            edge_probs = get_edge_distribution(gr, features,labels)

        for j, metric in enumerate(tqdm(metrics)):
            rules = range(0, Number_of_rules[dataset], rule_interval)
            #graph[-1].append(list())
            for rule in rules:
                graphs[-1][rule].append(list())
                eval = RuleEvaluator(model, dataset, datas=(gr, features, labels), target_rule=rule, edge_probs=edge_probs)
                #graphs[-1].append([list() for _ in metrics])
                tree = get_tree(dataset, rule, metric= metric,steps=epochs, nxp=nxp)
                for i, t in enumerate(tree):
                    nl = t.as_list()

                    nl = [el for el in nl if el.emb is not None]
                    #nl = [(el, el.value)
                    #      for el in nl if el.emb is not None]
                    # nl = get_pareto(nl)
                    #nl = sorted(nl, key=lambda x: x[1], reverse=True)[-1]
                    best = max(nl,key=lambda x: x.own_value)
                    graphs[-1][rule][j].append((best.graph,0))
                    #for g in nl:
                    #    graphs[-1][rule][j].append((g[0].graph,0))
        rules = range(0, Number_of_rules[dataset],rule_interval)

        for rule in tqdm(rules):
            #graphs[-1][rule].append(list())
            eval = RuleEvaluator(model, dataset, datas=(gr, features, labels), target_rule=rule, edge_probs=edge_probs)
            lay = eval.get_layer()
            support = list(filter(lambda x : eval.activate(x[0]), zip(embs[lay], egos[lay])))
            support = random.sample(support, min(10, len(support)))
            graphs[-1][rule].append(sorted([el for _, el in support],key=lambda x : len(x[0])))

            rnd = random.sample(egos[lay], len(support))
            graphs[-1][rule].append(sorted([el for el in rnd], key=lambda x : len(x[0])))

        values = list()
        #with graphs[-1] as ds :
        if True:
            ds=graphs[-1]
            #for m, ds in enumerate(graphs[-1]):
            values.append(list())
            for l, gxp in enumerate(tqdm(ds)):
                values[-1].append(np.zeros((len(metrics)+2,len(metrics)+2)))
                for i, g1 in enumerate(gxp):
                    for j, g2 in enumerate( gxp):
                        if i!=j:

                            for graph, r1 in g1:
                                graph1 = (
                                graph, [nx.Graph(graph.subgraph(list(graph.adj[node]) + [node])) for node in graph.nodes])

                                #v= max_common_subgraph(graph, g2, sample=False)
                                min_d = min([ged(graph1, gr) for gr,_ in g2])

                                values[-1][-1][ i, j]+= min_d/(len(graph)+len(graph.edges))
                            values[-1][-1][ i, j] /= len(g1)
        print(values)
        print(np.stack([el for x in values for el in x]).mean(axis=0))


def ged(g1, g2):
    """"
    using hungarian method
    """
    g2 = (g2, [nx.Graph(g2.subgraph(list(g2.adj[node]) + [node])) for node in g2.nodes])

    g1, nb1s = g1
    g2, nb2s = g2

    cost = np.ones((len(g1),len(g2)))

    for (i,ni), nb1 in zip(enumerate(g1.nodes), nb1s):
        for (j,nj),nb2 in zip(enumerate(g2.nodes), nb2s):
            cost[i,j]= nx.graph_edit_distance(nb1,nb2, node_match=lambda x,y:x["label"]==y["label"], roots =(ni,nj))
            if cost[i,j]!= cost[i,j]: # if nan because i,j are fixed it is impossible somitmes
                cost[i,j] = nx.graph_edit_distance(nb1,nb2, node_match=lambda x,y:x["label"]==y["label"])
    try:
        row_ind, col_ind = linear_sum_assignment(cost)
    except E:
        print(E, cost)
    return cost[row_ind, col_ind].sum()

def experiment_4_ratio_max_sg_df():

    #datasets = ["DD"]
    print("experiment_4_ratio_max_sg_df" + str(datasets))

    cols = ["dataset", "rule", "xp", "alpha","metr_name", "ismags"]

    metrics = ["cosine","entropy", "likelyhood_max"]
    nxp = 2
    epochs = 5000
    small= True
    alpha = [1 / (2 ** coef) for coef in range(10)] + [0.75] + [0]
    alpha = [coef/5 for coef in range(5)]+[1]

    for dataset in datasets:
        df = list()
        graphs, features, labels, _, _, _ = load_dataset(dataset)
        atom_map = get_atoms(dataset, (graphs, features, labels))

        dset = [dset_to_nx(g, x, atom_map=atom_map).subgraph(list(range(np.argmin(x.sum(axis=1))))) for g, x in zip(graphs, features)]

        for metric in metrics:
            for i, r in enumerate(alpha):
                rules = range(0, Number_of_rules[dataset])

                for rule in tqdm(rules):

                    tree = list()

                    if small :
                        for xp in range(nxp):
                            file = "results/mcts_dumps_new_small/dataset_" + dataset + "_rule_" + str(
                                rule) + "metric_" + metric + "xp_" + str(xp) + "steps_" + str(epochs) + "nxp_" + str(
                                nxp) + "ratio_" + str(r) + ".pkl"
                        with open(file, 'rb') as f1:
                            best= pickle.load(f1)[-1]

                            score = max_common_subgraph(best.graph, dset, sample=True)
                            df.append([dataset, rule, xp, r, metric, score])

                    else:
                        for xp in range(nxp):
                            #file = "results/mcts_dumps_ratio/dataset_" + dataset +"_rule_" + str(rule)+"metric_"+metric+"xp_" +str(xp)+"steps_"+str(epochs)+ "nxp_"+ str(nxp)+"ratio_"+str(r)+".pkl"
                            file = "results/mcts_dumps_new_small/dataset_" + dataset +"_rule_" + str(rule)+"metric_"+metric+"xp_" +str(xp)+"steps_"+str(epochs)+ "nxp_"+ str(nxp)+"ratio_"+str(r)+".pkl"

                            with open(file, 'rb') as f1:
                                tree.append(pickle.load(f1))

                        for xp, t in enumerate(tree):
                            # t.clean_tree()
                            nl = t.as_list()
                            #nl = get_pareto(nl)
                            nl = [el for el in nl if el.emb is not None]
                            best = sorted(nl, key= lambda x: x.own_value)[-1]

                            score = max_common_subgraph(best.graph, dset, sample=True, timer=0.2)
                            df.append([dataset, rule, xp, r,metric, score])


        df = pd.DataFrame(df,
            columns=cols)
        df.to_csv("results/"+out_dir+"/dataframes/mcts_ratio_common_sg_"+ dataset+".csv")



def experiment_4_ratio():
    datasets = ["aids"]  # , "mutag", "BBBP", "DD"]


    metrics = ["cosine", "entropy", "likelyhood_max"]

    convolve = 50
    confidence = 0.95
    tval = norm.ppf((1 + confidence) / 2)
    # rules = range(0,Number_of_rules[dataset], 5)

    dis = "min"
    mark  =["o", "x", "+","1","2","3","*"]
    for dataset in datasets:
        rules = range(0, Number_of_rules[dataset], 5)

        for metric in metrics:
            for i, r in enumerate([1 / 8, 1 / 4, 1 / 2, 1., 2.]):
                X = list()
                Y = list()

                for rule in tqdm(rules):
                    with open("results/mcts_dumps_ratio/dataset_" + dataset + "_rule_" + str(
                        rule) + "metric_" + metric + "xp_" + str(0) + "raito_" + str(r) + ".pkl", 'rb') as f1:
                        x, y = pickle.load(f1)
                    X += x
                    Y += y
                plt.plot(x, y, marker=mark[i],label=str(r))
            plt.legend()
            #plt.title(metric)

            #plt.savefig("results/mcts_egos2/pareto" + "_dataset_" + dataset + "_rule_" + str(rule) + ".png")
            plt.show()



def experiment_4_ratio_df():
    datasets = ["mutag","BBBP", "aids", "DD" ,"PROTEINS_full"]

    cols = ["dataset", "rule", "xp", "step", "alpha", "layer", "metr_name", "metr_val","metr_score", "metr_real"] + ["c_" + str(i) for i in range(20)]

    metrics = ["cosine","entropy", "likelyhood_max"]
    nxp = 2
    epochs = 5000

    small=True

    alpha = [1 / (2 ** coef) for coef in range(10)] + [0.75] + [0]
    alpha = [coef / 5 for coef in range(5)] + [1]
    for dataset in datasets:
        df = list()
        graphs, features, labels, _, _, _ = load_dataset(dataset)
        if dataset =="PROTEINS_full":
            edge_probs = get_edge_distribution(graphs, features,labels, 30)
        else :
            edge_probs = get_edge_distribution(graphs, features,labels)


        for metric in metrics:
            for i, r in enumerate(alpha):
                rules = range(0, Number_of_rules[dataset])

                for rule in tqdm(rules):

                    tree = list()
                    if small :
                        for xp in range(nxp):
                            file = "results/mcts_dumps_new_small/dataset_" + dataset + "_rule_" + str(
                                rule) + "metric_" + metric + "xp_" + str(xp) + "steps_" + str(epochs) + "nxp_" + str(
                                nxp) + "ratio_" + str(r) + ".pkl"
                        with open(file, 'rb') as f1:
                            best= pickle.load(f1)[-1]

                            df.append([dataset, rule, i, best.step, r, best.layer, metric, best.own_value,
                                       best.own_value_detailed[0],
                                       best.own_value_detailed[1]] + best.emb.tolist())
                    else:
                        for xp in range(nxp):
                            file = "results/mcts_dumps_ratio/dataset_" + dataset +"_rule_" + str(rule)+"metric_"+metric+"xp_" +str(xp)+"steps_"+str(epochs)+ "nxp_"+ str(nxp)+"ratio_"+str(r)+".pkl"


                            with open(file, 'rb') as f1:
                                tree.append(pickle.load(f1))

                        for i, t in enumerate(tree):
                            # t.clean_tree()
                            nl = t.as_list()
                            #nl = get_pareto(nl)
                            nl = [el for el in nl if el.emb is not None]
                            best = sorted(nl, key= lambda x: x.own_value)[-1]

                            scores = []
                            df.append([dataset, rule, i, best.step, r, best.layer, metric, best.own_value, best.own_value_detailed[0],
                                       best.own_value_detailed[1]] + scores + best.emb.tolist())


        df = pd.DataFrame(df,
            columns=cols)
        df.to_csv("results/"+out_dir+"/dataframes/ratio_"+ dataset+".csv")

def experiment_4_ratio_xgnn_df(random=True):
    #datasets = ["DD", "PROTEINS_full"]  # "", "mutag", "BBBP", "DD"]

    cols = ["dataset", "rule", "xp", "step", "alpha", "layer", "metr_name", "metr_val","metr_score", "metr_real", "ismags"] + ["c_" + str(i) for i in range(20)]

    metrics = ["cosine", "entropy","likelyhood_max"]
    if random:
        metrics =["cosine"]
    vals = list()

    alpha = [1 / (2 ** coef) for coef in range(10)] + [0.75] + [0]
    alpha = [coef / 5 for coef in range(5)] + [1]

    if random == True:
        alpha = [0.5]
    for j, dataset in enumerate(datasets):
        rules = range(0, Number_of_rules[dataset])
        grs = list()
        graphs, features, labels, _, _, _ = load_dataset(dataset)
        datas = list(zip(graphs, features, labels))
        atom_map = get_atoms(dataset, (graphs, features, labels))

        dset = [dset_to_nx(g, x, atom_map=atom_map).subgraph(list(range(np.argmin(x.sum(axis=1))))) for g, x in zip(graphs, features)]


        if dataset == "PROTEINS_full":
            edge_probs = get_edge_distribution(graphs, features,labels, 30)
        else:
            edge_probs = get_edge_distribution(graphs, features,labels)
        model, _ = model_selector("GNN", dataset, pretrained=True, return_checkpoint=True)
        df = list()
        for metric in metrics:
            for i, ratio in enumerate(alpha):

                for rule in tqdm(rules):
                    cur = list()
                    evaluator = RuleEvaluator(model, dataset, datas=(graphs, features, labels), target_rule=rule,
                                              metric=metric, edge_probs=edge_probs)
                    if random == True :
                        budget = 500
                        nxp= 2
                        for xp in range(nxp):
                            file = "results/random_dumps_new/dataset_" + dataset + "_rule_" + str(
                                rule) + "metric_" + metric + "xp_" + str(xp) + "steps_" + str(budget) + "nxp_" + str(
                                nxp) + "ratio_" + str(ratio) + "random.pkl"
                            with open(file, "rb") as f:
                                cur.append(pickle.load(f)[-1])
                    else :
                        nxp = 2
                        budget = 5000
                        for xp in range(nxp):
                            file = "results/xgnn_dumps_new/dataset_" + dataset + "_rule_" + str(
                                rule) + "metric_" + metric + "xp_" + str(xp) + "steps_" + str(budget) + "nxp_" + str(
                                nxp) + "ratio_" + str(ratio) + ".pkl"
                            with open(file, "rb") as f:
                                cur.append(pickle.load(f)[-1])


                    if False :
                        cur = cur[0][:min(10, len(cur))]#TODO
                    else :
                        for xp, el in enumerate(cur):
                            emb = evaluator.get_embedding(el[-1], evaluator.get_layer())

                            value = evaluator.compute_score_emb(emb, metric)
                            real = evaluator.real_score(el[-1])
                            max_sg = max_common_subgraph(el[-1], dset, sample=True,timer=0.2)

                            #cur = sorted([(gr, evaluator.compute_score(gr, metric))  for curxp in cur for gr in curxp], key= lambda x: x[1], reverse=True)[:min(10, len(cur))]
                            df.append([dataset, rule, xp, 0, ratio, 0, metric, 0,
                                       value,
                                       real, max_sg] + [] + emb.tolist())

                    #for c in tqdm(cur):
                    #    vals.append([dataset, rule, 0, -1,  metric,  np.max(max_common_subgraph(c, datas))])
                    #vals.append([np.max(max_common_subgraph(c, datas)) for c in cur])


        df = pd.DataFrame(df,
            columns=cols)
        if random==True:
            df.to_csv("results/"+out_dir+"/dataframes/ratio_random"+ dataset+".csv")
        else :
            df.to_csv("results/"+out_dir+"/dataframes/ratio_xgnn"+ dataset+".csv")


def compute_value_base(eval, emb, graph, metric, beta=1):
    val = eval.compute_score_emb(emb, metric)
    real  = eval.real_score(graph)

    mi_real = -20
    ma_rear = -2

    score = (val - real_ratio[metric][0]) / (
                real_ratio[metric][1] - real_ratio[metric][0]) + beta * (real - mi_real) / (ma_rear - mi_real)
    return score


def compute_value(eval, node, metric, beta=1):
    return compute_value_base(eval, node.emb, node.graph, metric, beta)



def experiment_4_ratio_plots():
    print("experiment_4_ratio_plots")
    #plt.figure(figsize=(20, 15))

    #datasets = ["DD"]
    cols = ["dataset", "rule", "xp", "step", "layer", "metr_name", "metr_val","metr_score", "metr_real"] + ["c_" + str(i) for i in range(20)]

    plt.rcParams.update({'font.size': 12})
    metrics = ["cosine", "entropy","likelyhood_max"]
    #metrics = ["cosine"]

    rename_metrics = {"cosine": "Cosine", "entropy": "Cross-Entropy", "likelyhood_max" :"Relative-CE"}

    alpha = [1 / (2 ** coef) for coef in range(10)]+[0]
    stiles = [":","--", "-","-."]
    confidence = 0.99
    tval = norm.ppf((1 + confidence) / 2)

    metric = "cosine"
    metr_score = "ismags"
    methods = ["mcts_ratio_common_sg_", "ratio_xgnn"]
    method_name = {"mcts_ratio_common_sg_": "Discern", "ratio_xgnn":"XGNN++"}
    for i, dataset in enumerate(datasets):
        plt.figure(figsize=(10, 7))

        for method in methods:

            for j, metric in enumerate(metrics):
                    # rules = range(Number_of_rules[dataset])
                    # for r in rules:

                    f ="results/"+out_dir+"/dataframes/"+method+""+ dataset+".csv"
                    df = pd.read_csv(f)

                    vals = df[(df["metr_name"] == metric)].groupby("alpha")[metr_score]

                    nels = (df["metr_name"] == metric).sum() / len(vals)
                    values = vals.mean()
                    #values = ((values -real_ratio[metric][0])/ (real_ratio[metric][1] -real_ratio[metric][0])).tolist()
                    values.tolist()
                    alpha = vals.mean().index.tolist()
                    std = [0 for _ in vals]#tval*np.array(vals.std())/ np.sqrt(nels)

                    #v = np.convolve(values, np.ones(convolve), "valid") / convolve
                    #std = np.convolve(np.array(std) * tval, np.ones(convolve), "valid") / convolve

                    #plt.errorbar(alpha, values, std, linestyle=stiles[j], label=m, elinewidth=1)
                    plt.plot(alpha, values, linestyle=stiles[j], label=method_name[method]+" "+rename_metrics[metric])
                    #plt.plot(alpha, values, linestyle=stiles[j], label=rename_metrics[metric], elinewidth=1)

                    plt.xlabel("Beta")

        metric= "cosine"

        f = "results/" + out_dir + "/dataframes/ratio_random" + dataset + ".csv"
        df = pd.read_csv(f)

        vals = df[(df["metr_name"] == metric)]["metr_score"]
        nels = (df["metr_name"] == metric).sum() / len(vals)

        # vals = np.array([[compute_value_base(evaluator, emb, gr, metric, beta=al) for gr, emb in grs] for al in alpha])
        values = vals.mean().tolist()

        std = tval * np.array(vals.std()) / np.sqrt(len(vals))
        plt.plot([0, 1], [1, 1], label="DISC-GSPAN")

        plt.plot([0,0.5, 1], [values, values,values], label="random")

        # plt.errorbar(alpha, values, std, linestyle=stiles[3], label="random", elinewidth=1)
        # plt.xlabel("Beta")
        """for j, metric in enumerate(["cosine", "entropy"]):
            f = "results/"+out_dir+"/dataframes/ratio_xgnn" + dataset + ".csv"
            df = pd.read_csv(f)

            vals = df[(df["metr_name"] == metric)]["metr_val"]
            nels = (df["metr_name"] == metric).sum() / len(vals)
            values = vals.mean().tolist()

            std = tval * np.array(vals.std()) / np.sqrt(len(vals))
            plt.errorbar([0, 2, 4], [values, values, values], [0, std, 0],linestyle=stiles[j], label="xgnn "+metric)"""

        #plt.title("Realism "+ dataset)
        plt.legend(loc='lower right')

        #fig.tight_layout()
        plt.savefig("results/figs/realism_" + dataset + ".png")

        plt.show()


def xp_graph_metric_mcts():
    print("6 xp_graph_metric")
    datasets =["mutag"]#, "mutag", "BBBP", "DD"]


    rules = [23,28 ]
    nxp=2
    metrics = ["cosine", "entropy","likelyhood_max"]
    for dataset in datasets :
        graphs, features, labels, _, _, _ = load_dataset(dataset)
        if dataset =="PROTEINS_full":
            edge_probs = get_edge_distribution(graphs, features,labels,30)
        else :
            edge_probs = get_edge_distribution(graphs, features,labels)

        for j, metric in enumerate(metrics):
            atoms = get_atoms(dataset,None)

            for rule in rules:
                #eval = RuleEvaluator(None, dataset, (datas), rule, [metric])
                evaluator = RuleEvaluator(None, dataset, datas=(graphs, features, labels), target_rule=rule,
                                          metric=metric, edge_probs=edge_probs)

                for xp in range(nxp):

                    """file = "results/xgnn_dumps_new/dataset_" + dataset + "_rule_" + str(
                            rule) + "metric_" + metric + "xp_" + str(i) + "steps_" + str(budget) + "nxp_" + str(
                            nxp) + ".pkl" """
                    file = "results/mcts_dumps_new_small/dataset_" + dataset + "_rule_" + str(
                            rule) + "metric_" + metric + "xp_" + str(xp) + "steps_" + str(epochs) + "nxp_" + str(
                            nxp) + "ratio_" + str(0.4) + ".pkl"

                    with open(file, "rb") as f:
                        cur = pickle.load(f)

                        grs = cur[-1]
                        #grs = cur[-1]
                        # for i, gr in enumerate(grs):
                        # for i in gr.nodes:
                        #    gr.nodes[i]["label"] = evaluator.atoms[gr.nodes[i]["label"]]
                        #grs = [(graph, evaluator.compute_score(graph, metric)) for graph in grs]
                        #grs = sorted(grs, key=lambda x: x[1], reverse=True)[:1]
                    #for i, (gr, qual) in enumerate(grs):

                    title = dataset + " " + str(rule) + " " + metric + " "+ str(xp)+ " " + str(grs.own_value)
                    graph_draw(grs.graph)
                    plt.savefig("results/ego_plots/mcst" + title + ".png")

                    plt.show()

def xp_graph_metric_gpan():
    print("6 xp_graph_metric gspan")
    datasets = ["mutag", "mutag", "BBBP"]

    rules = [23, 28]
    metrics = [""]
    for dataset in datasets:
        graphs, features, labels, _, _, _ = load_dataset(dataset)
        if dataset == "PROTEINS_full":
            edge_probs = get_edge_distribution(graphs, features,labels ,30)
        else:
            edge_probs = get_edge_distribution(graphs, features,labels)

        for j, metric in enumerate(metrics):
            atoms = get_atoms(dataset, None)

            for rule in rules:
                # eval = RuleEvaluator(None, dataset, (datas), rule, [metric])
                evaluator = RuleEvaluator(None, dataset, datas=(graphs, features, labels), target_rule=rule,
                                          metrics=[], edge_probs=edge_probs)
                file = "results/gspan/gspan_" + dataset + "_rule_" + str(rule) + ".pkl"

                with open(file, "rb") as f:
                    try:
                        result = pickle.load(f)
                    except Exception:
                        return
                graph  = result["graph"]
                val = result["value"]
                gg= nx.from_numpy_matrix(graph[0].numpy())
                for i in gg.nodes:
                    gg.nodes[i]["label"] = evaluator.atoms[np.argmax(graph[1][i]).item()]

                title = dataset + " " + str(rule)+"wrac_"+str(val)
                graph_draw(gg, noCenter=True)
                plt.savefig("results/ego_plots/gspan"+title+".png")
                plt.show()

def xp_graph_metric_xgnn(random=False):
    print("6 xp_graph_metric gspan")
    datasets = ["mutag"]  # , "mutag", "BBBP", "DD"]
    r=random
    rules = [23, 28]
    budget=5000
    nxp=2
    random = False
    metrics = ["cosine", "entropy","likelyhood_max"]
    for dataset in datasets:
        graphs, features, labels, _, _, _ = load_dataset(dataset)
        if dataset == "PROTEINS_full":
            edge_probs = get_edge_distribution(graphs, features,labels ,30)
        else:
            edge_probs = get_edge_distribution(graphs, features,labels)

        for j, metric in enumerate(metrics):
            atoms = get_atoms(dataset, None)

            model, _ = model_selector("GNN", dataset, pretrained=True, return_checkpoint=True)

            for rule in rules:
                # eval = RuleEvaluator(None, dataset, (datas), rule, [metric])
                evaluator = RuleEvaluator(model, dataset, datas=(graphs, features, labels), target_rule=rule,
                                          metric=metric, edge_probs=edge_probs)


                for xp in range(nxp):
                    if r:
                        file = "results/random_dumps_new/dataset_" + dataset + "_rule_" + str(
                            rule) + "metric_" + metric + "xp_" + str(xp) + "steps_" + str(budget) + "nxp_" + str(
                            nxp) + "random.pkl"
                    else:
                        """file = "results/xgnn_dumps_new/dataset_" + dataset + "_rule_" + str(
                            rule) + "metric_" + metric + "xp_" + str(xp) + "steps_" + str(budget) + "nxp_" + str(
                            nxp) + ".pkl" """
                        file = "results/xgnn_dumps_new/dataset_" + dataset + "_rule_" + str(
                            rule) + "metric_" + metric + "xp_" + str(xp) + "steps_" + str(budget) + "nxp_" + str(
                            nxp) + "ratio_" + str(0.4) + ".pkl"

                    with open(file, "rb") as f:
                        cur = pickle.load(f)

                        grs = cur[-1]
                        #for i, gr in enumerate(grs):
                            #for i in gr.nodes:
                            #    gr.nodes[i]["label"] = evaluator.atoms[gr.nodes[i]["label"]]
                        grs = [(graph, evaluator.compute_score(graph, metric)) for graph in grs]
                        grs = sorted(grs, key=lambda x: x[1], reverse=True)[:1]
                    for i,  (gr, qual) in enumerate(grs):

                        title = dataset + " " + str(rule) + " " + metric + " " + str(xp) +" " +str(qual)
                        graph_draw(gr)
                        if r :
                            plt.savefig("results/ego_plots/random" + title + ".png")
                        else:
                            plt.savefig("results/ego_plots/xgnn"+title+".png")
                        plt.show()



def graph_draw(graph, noCenter=False):
    cols = defaultdict(lambda: "y")
    cols.update({0: 'g', 1: 'r', 2: 'b', 3: 'c', 4: 'm', 5: 'w', 6: 'y'})

    attr = nx.get_node_attributes(graph, "label")
    labels = {}
    color = ''

    for n in attr:
        labels[n] = attr[n]
        color = color + cols[attr[n]]
    color = list(color)
    if not noCenter:
        color[0] = "r"
    #   labels=dict((n,) for n in attr)
    nx.draw_kamada_kawai(graph, labels=labels, node_color=color)
    #nx.draw(graph, labels=labels, node_color=color)




def gspan_embedding(dataset, rule, eval):
    file = "results/gspan/gspan_" + dataset + "_rule_" + str(rule)+".pkl"
    with open(file,"rb") as f:
        try:
            result = pickle.load(f)
        except Exception:
            return []
    graph = result["graph"]
    adj, feat = graph[0], graph[1]
    adj = dense_to_sparse(adj)[0]
    embeddinng = eval.gnnNets.embeddings(feat, adj)[eval.get_layer()]
    #emb = eval.get_embedding_adj(graph[0], graph[1], eval.get_layer())

    emb = embeddinng[0]
    target= target_embedding(None, rule, eval)
    min = ((emb -target[1])**2).sum().detach()
    for e in embeddinng:
        cur = ((e - target[1]) ** 2).sum().detach()
        if cur <min :
            min = cur
            emb = e

    return [torch.clamp(emb, 0, 1)]

    #return [torch.clamp(emb, 0, 1)]

    #return [torch.clamp(embeddinng, 0, 1)]


def xgnn_embedding(dataset, rule,metric, eval, random=False):
    nxp= 3
    budget=5000
    r= random
    ratio= 0.5
    result=list()
    for i in range(nxp):
        if r:
            file = "results/random_dumps_new/dataset_" + dataset + "_rule_" + str(
                rule) + "metric_" + metric + "xp_" + str(i) + "steps_" + str(budget) + "nxp_" + str(
                nxp) + "ratio_"+str(ratio)+"random.pkl"
        else:
            file = "results/xgnn_dumps_new/dataset_" + dataset + "_rule_" + str(
                rule) + "metric_" + metric + "xp_" + str(i) + "steps_" + str(budget) + "nxp_" + str(
                nxp) + "ratio_"+str(ratio)+".pkl"

        with open(file , "rb") as f:
            cur = pickle.load(f)
            grs= [ graph for graph in cur[0][-5:]]

            result += [eval.get_embedding(g, eval.get_layer()) for g in grs]
    return result

def mcts_embedding(dataset, rule,metric, eval, small=False):
    epochs = 5000
    nxp=6
    ratio =0.5
    if dataset == "PROTEINS_full":
        nxp = 6
    result = list()

    if small:
        for xp in range(nxp):
            file = "results/" + in_dir + "_small/dataset_" + dataset + "_rule_" + str(
                rule) + "metric_" + metric + "xp_" + str(xp) + "steps_" + str(epochs) + "nxp_" + str(
                nxp) + "ratio_" + str(ratio) + ".pkl"
            with open(file, "rb") as f1:
                val = pickle.load(f1)
            best = val[-1]
            result+=[torch.clamp(best.emb, 0, 1)]
    else :
        tree = get_tree(dataset, rule, metric=metric, steps=epochs, nxp=nxp)
        result = list()
        for i, t in enumerate(tree):
            nl= t.as_list()
            nl = [(el.emb, compute_value(eval, el, metric, beta=1)) for el in nl if el.emb is not None]
            nl = sorted(nl, key=lambda x: x[1], reverse=True)[:1]
            result+=[torch.clamp(el, 0, 1) for el,_ in nl]
    return result

def max_likelyhood(embedding, rule):

    target_rule, _,rules = rule
    emb = torch.clamp(embedding, 0, 1)
    #rules = [el for el in rules if el[0] == target_rule[0]]
    #target_id = rules.index(target_rule)
    probs = [(emb[rule[1]] + 1E-5).log().sum().item() for rule in rules if rule!=target_rule]
    #probs = [self.log_likelyhood(emb, rule) for rule in rules]

    return (emb[target_rule] + 1E-5).log().sum().item() - max(probs)


def target_embedding(dataset, rule, eval):
    out = torch.zeros((20))
    out[eval.target_rule[1]]=1
    rules = [el for el in eval.rules if el[0] == eval.rules[rule][0]]
    return [eval.target_rule[1],out, rules]




def eval_method(embedding, target, metric):
    values = [metric(emb, target) for emb in embedding]
    return values#, np.min(values), np.mean(values)
from torch.nn.functional import cosine_similarity


def xp_5_competitors_l2(small=False):
    print("xp_5_competitors")

    #datasets = ["aids" ,"BBBP", "mutag"]#, "DD", "PROTEINS_full","ba2" ]# ''"DD"]

    datasets= ["ba2"]
    methods  = ["cosine","entropy", "likelyhood_max"]
    metrics = {"l2" : lambda x, y : ((x -y[1])**2).sum().detach()}
    """,
               "cosine": lambda x, y: cosine_similarity(x,y[1],dim=0).item(),
                "entropy": lambda x, y:(x[y[0]] + 1E-5).log().sum().item(),
               "likelyhood_max" : max_likelyhood
               }"""
    confidence = 0.95
    tval = norm.ppf((1 + confidence) / 2)
    # rules = range(0,Number_of_rules[dataset], 5)
    target = list()
    results = {metr:{dset :{ } for dset in datasets} for metr in metrics.keys()}
    results_std = {metr:{dset :{ } for dset in datasets} for metr in metrics.keys()}
    method_names = list()
    for d_id, dataset in enumerate(datasets):
        embeddings = defaultdict(list)

        target = list()


        gr, features, labels, _, _, _ = load_dataset(dataset)

        if dataset =="PROTEINS_full":
            edge_probs = get_edge_distribution(gr, features,labels,30)
        else :
            edge_probs = get_edge_distribution(gr, features,labels)
        embs, model, datas = get_dset_embeddings(dataset)

        rules = range(0, Number_of_rules[dataset])

        for rule in tqdm(rules):

            eval = RuleEvaluator(model, dataset, datas, rule, [],edge_probs=edge_probs)

            #embeddings["gspan"].append(gspan_embedding(dataset, rule,eval))
            for method in methods:
                embeddings["xgnn_"+method].append(xgnn_embedding(dataset, rule, method, eval, random=False))
                embeddings["random_"+method].append(xgnn_embedding(dataset, rule, method, eval, random=True))

                #embeddings["mcts_"+method].append(mcts_embedding(dataset, rule, method, eval, small))

            target.append(target_embedding(dataset, rule, eval))
        method_names = list(embeddings.keys())



        for method_id, (k, embs) in enumerate(embeddings.items()):
            res = [list() for _ in metrics]
            for i, metr in enumerate(metrics):
                res = [eval_method(e, t, metrics[metr]) for e, t in zip(embs, target)]
                res = [el for xx in res for el in xx]
                results[metr][dataset][k] = np.mean(res)
                results_std[metr][dataset][k] =np.std(res)

    print(method_names)

    for k in results_std.keys():
        print(k)
        for method in method_names:
            #line = method + " "
            vals=list()
            std =list()
            for dataset in datasets:
                vals.append(results[k][dataset][method])
                std.append(results_std[k][dataset][method])
                #line +=  " &"+  str(results[k][dataset][method]) + " \pm " +str(results_std[k][dataset][method]) +" "
            latex_metrics(vals,std, method)
            #print(line+ "\\\\\\hline")


#datasets = ["DD", "aids", "mutag",  "BBBP" , "PROTEINS_full", "ba2"]
#datasets = ["ba2"]
datasets = ["DD", "aids", "mutag",  "BBBP" , "PROTEINS_full", "ba2"]

#xp_5_competitors_l2()

#xp_graph_metric()"
#to_dataframe(small=False)
datasets = ["DD", "aids", "mutag",  "BBBP" , "PROTEINS_full", "ba2"]

experiment_learnig_curve_plot()
#
#test_similarities()

#to_dataframe()

#experiment_2_quartiles()

#experiment_3_l2()

#experiment_3_bis()

#experiment_4_ratio()
#datasets = ["aids", "mutag",  "BBBP", "DD" , "PROTEINS_full"]
#experiment_4_ratio_df()
#experiment_4_ratio_max_sg_df()
#experiment_4_ratio_xgnn_df(random=False)
#experiment_4_ratio_xgnn_df(random=True)
#experiment_4_ratio_plots()

#xp_5_competitors_l2(small=True)



#xp_graph_metric()

#xp_graph_metric_xgnn(random=True)

#xp_graph_metric_xgnn()
#xp_graph_metric_gpan()
#xp_graph_metric_mcts()



#to_dataframe_bis()
#experiment_learnig_curve_plot_bis()



