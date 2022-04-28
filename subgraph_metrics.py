import random

import torch

from networkx.algorithms.isomorphism import ISMAGS

import pandas as pd
from tqdm import tqdm
from ExplanationEvaluation.datasets.dataset_loaders import load_dataset

import pysubgroup as ps
from torch_geometric.utils import to_dense_adj, dense_to_sparse

import numpy as np
import networkx as nx

import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from collections import defaultdict
datasets = ["mutag"]

import json

def get_metric(dataset, metric):
    """compares with dataset"""
    graphs, features, labels, _, _, _ = load_dataset(dataset)
    datas = list(zip(graphs, features, labels))

    datas =[dset_to_nx(g,x).subgraph(list(range(np.argmin(x.sum(axis=1))))) for g,x in zip(graphs,features)]

    motifs = read_subgroup(dataset)
    out =list()
    for motif in motifs:
        out.append(metric(motif, datas))#(graphs, features,labels)))
    return out

def compare_motifs(dataset, metric):
    xgnn_motifs = read_subgroup(dataset)
    inside_motifs = read_inside_motifs(dataset)
    #graphs, features, labels, _, _, _ = load_dataset(dataset)
    #datas = list(zip(graphs, features, labels))


    out =list()
    for motif in xgnn_motifs:
        out.append(metric(motif, inside_motifs))#"(graphs, features,labels)))
    out = np.array(out)
    return out

def compare_motifs_with_set(dataset, graph_list, metric, rule_eval):
    graphs, features, labels, _, _, _ = load_dataset(dataset)
    datas = list(zip(graphs, features, labels))

    datas = [dset_to_nx(g, x, rule_eval.atoms).subgraph(list(range(np.argmin(x.sum(axis=1))))) for g, x in zip(graphs, features)]


    out = list()
    for motif in graph_list:
        out.append(metric(motif, datas))  # (graphs, features,labels)))
    return out


def dset_to_nx(graph, features, atom_map=None):
    dense_adj = to_dense_adj(torch.tensor(graph)).numpy()[0]

    g = nx.from_numpy_matrix(dense_adj)  # nx.Graph()
    if atom_map is not None:
        g.add_nodes_from([(i, {"label": atom_map[el]}) for i, el in enumerate(np.argmax(features, axis=1))])
    else:
        g.add_nodes_from([(i, {"x": el}) for i, el in enumerate(np.argmax(features, axis=1))])

    # g.add_edges_from(graph.transpose().tolist())
    return g

def read_inside_motifs(dataset):
    with open("results/"+dataset+"_results.json","r") as f:
        data = json.load(f)
    graphs = [0]*60
    target = "results/egos/"+dataset
    for g in data:
        if g["dataset_name"][:len(target)]==target:
            index = int(g["dataset_name"][len(target)+1:].split("l")[0])
            graphs[index] =parse_graph(g["graph"])

    return graphs

def parse_graph(text):
    data = text.split(" ")
    x= data[0]
    i= 1
    nodes = list()
    edges  =list()
    while x=="v":
        n, l, x =data[i:i+3]
        nodes.append((int(n),{"x": int(l)}))

        i+=3
    while i< len(data)-1:
        s, e, l,x =data[i:i+4]
        edges.append((int(s),int(e)))

        i+=4
    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    return g

def get_subgraph_cover(graph, dataset):
    n= 0
    for g in dataset:
        n+=ISMAGS(g, graph, node_match="label")

    return n




def read_subgroup(dataset):
    names = {"ba2": ("ba2"),
             "aids": ("Aids"),

             "mutag": ("Mutagenicity"),
             }
    name = names[dataset]
    filename = "results/"+ name + "_graph_XGNN.txt"
    out = list()
    with open(filename, "r") as f:
        l = f.readline()
        while True:
            while l[0]!="t":
                l = f.readline()
                if not l:
                    return out

            cls = int(l.split("#")[-1])
            if cls ==-1 :
                return out
            g, l =parse_subgraph(f)
            #out.append((g,cls))
            out.append(g)

            #while l!="\n":

    return out

def parse_subgraph(fp):
    l= fp.readline()

    nodes = list()
    edges = list()
    while l[0]!= "e":
        _, n, label = l.split(" ")
        nodes.append((int(n),{"x": int(label)}))
        l = fp.readline()

    while l[0]!= "t":
        n1,n2= l.split(" ")[1:3]
        edges.append((int(n1),int(n2)))

        l = fp.readline()

    #nodes = [(el, {"x":kwargs["atoms"][cl]}) for (el,cl),_ in lines]
    #edges = [(el[0], nb) for el,nbs in lines for nb in nbs]
    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    return g.to_undirected(),l

from tqdm import tqdm
from networkx.algorithms.similarity import optimize_graph_edit_distance
from networkx.algorithms.isomorphism import ISMAGS

def common_subgraph(graph, dset):
    is_subgraph=list()
    max_subgraph=list()

    #graph2 = (graph, [nx.Graph(graph.subgraph(list(graph.adj[node]) + [node])) for node in graph.nodes])
    for G in random.sample(dset, 100):

        #d.append(ISMAGS(G, graph,node_match=lambda x,y:x["x"]==y["x"]).subgraph_is_isomorphic())
        iso = ISMAGS(G, graph, node_match=lambda x, y: x["label"] == y["label"])
        is_subgraph.append(iso.subgraph_is_isomorphic())

        max_subgraph.append(list(iso.largest_common_subgraph()))


        #d.append( ged(graph2,G))#,node_match=lambda x,y:x["x"]==y["x"])
    return [len(el[0])/len(graph) if len(el) else 0 for el in max_subgraph]

def signal_handler(signum, frame):
    raise Exception("Timed out!")
import signal

def max_common_subgraph(graph, dset, sample=True, timer=None):
    is_subgraph = list()
    max_subgraph = list()

    # graph2 = (graph, [nx.Graph(graph.subgraph(list(graph.adj[node]) + [node])) for node in graph.nodes])
    max = 0
    if sample:
        iter = random.sample(dset, 100)
    else :
        iter = dset
    for G in iter:
        # d.append(ISMAGS(G, graph,node_match=lambda x,y:x["x"]==y["x"]).subgraph_is_isomorphic())

        iso = ISMAGS(G, graph, node_match=lambda x, y: x["label"] == y["label"])
        #is_subgraph.append(iso.subgraph_is_isomorphic())
        el=None

        #max_subgraph.append(list(iso.largest_common_subgraph()))
        signal.signal(signal.SIGALRM, signal_handler)
        if timer :
            signal.setitimer(signal.ITIMER_REAL,timer)
        else :
            signal.alarm(0)
        try:
            #for i in iso.largest_common_subgraph():
            el = next(iso.largest_common_subgraph(), None)
            #if (node in i.keys()):
            #    el.append(i)
        except Exception as e:
            pass
        signal.alarm(0)
        #el = list(iso.largest_common_subgraph())
        if el is not None and len(el) / len(graph) >max:
            max = len(el) / len(graph)
            if max >= 1:
                return 1
        """ if len(el) is not None and len(el[0]) / len(graph) >max:
            max= len(el[0]) / len(graph)
            if max >= 1:
                return 1"""
        # d.append( ged(graph2,G))#,node_match=lambda x,y:x["x"]==y["x"])
    return max

def exact_ged(graph, dset):
    """
    but "slow"
    """
    d=list()

    for G in tqdm(dset):
        out =nx.graph_edit_distance(graph,G, node_match=lambda x,y:x["x"]==y["x"])
        d.append(out)
        #d.append( ged(graph2,G))#,node_match=lambda x,y:x["x"]==y["x"])
    return d

def ged_metric(graph, dset):
    d=list()

    graph2 = (graph, [nx.Graph(graph.subgraph(list(graph.adj[node]) + [node])) for node in graph.nodes])
    for G in tqdm(dset):
        d.append( ged(graph2,G))#,node_match=lambda x,y:x["x"]==y["x"])
    return d


def ged(g1, g2):
    """"
    using hungarian method
    """
    g2 = (g2, [nx.Graph(g2.subgraph(list(g2.adj[node]) + [node])) for node in g2.nodes])

    g1, nb1s = g1
    g2, nb2s = g2

    cost = np.ones((len(g1),len(g2)))
    for i, nb1 in zip(g1.nodes, nb1s):
        for j,nb2 in zip(g2.nodes, nb2s):
            cost[i,j]= nx.graph_edit_distance(nb1,nb2, node_match=lambda x,y:x["x"]==y["x"], roots =(i,j))
            if cost[i,j]!= cost[i,j]: # if nan because i,j are fixed it is impossible somitmes
                cost[i,j] = nx.graph_edit_distance(nb1,nb2, node_match=lambda x,y:x["x"]==y["x"])
    try:
        row_ind, col_ind = linear_sum_assignment(cost)
    except E:
        print(E, cost)
    return cost[row_ind, col_ind].sum()

def graph_draw(graph):
    atoms = {0: "C", 1: "O", 2: "Cl", 3: "H", 4: "N", 5: "F", 6: "Br", 7: "S", 8: "P", 9: "I", 10: "Na", 11: "K", 12: "Li", 13: "Ca"}

    color = defaultdict(lambda: "y")
    color.update({0: 'g', 1: 'r', 2: 'b', 3: 'c', 4: 'm', 5: 'w', 6: 'y'})
    attr = nx.get_node_attributes(graph, "x")
    labels = {}
    col = ''
    for n in attr:
        labels[n] = atoms[attr[n]]
        col = col + color[attr[n]]

    #   labels=dict((n,) for n in attr)
    nx.draw_kamada_kawai(graph, labels=labels)

"""
planarity("mutag")
#get_metric("mutag", common_subgraph)
compare_motifs("mutag", exact_ged)
#compare_motifs("mutag", ged_metric)
compare_motifs("mutag", common_subgraph)
"""