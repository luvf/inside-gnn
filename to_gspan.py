
import pandas as pd
from tqdm import tqdm
from ExplanationEvaluation.datasets.dataset_loaders import load_dataset

import pysubgroup as ps
from torch_geometric.utils import to_dense_adj, dense_to_sparse

import numpy as np
import networkx as nx

import matplotlib
import matplotlib.pyplot as plt
from ExplanationEvaluation.models.model_selector import model_selector
from ExplanationEvaluation.explainers.utils import RuleEvaluator

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


def load_activations(dataset_name):
    model = "GNN"
    explainer  = "ACT"
    file = 'results/trace/'+dataset_name+"_"+model+"_"+explainer+"_activation.csv"
    df = pd.read_csv(file)
    return df


def load_motifs(dataset):
    names= {"ba2": ("ba2"),
            "aids" : ("Aids"),
            "BBBP": ("Bbbp"),
            "mutag": ("Mutag"),
            "DD": ("DD"),
            "PROTEINS_full": ("Proteins")}
    name = names[dataset]
    file = "ExplanationEvaluation/datasets/activations/"+ name+"/"+name+"_activation_encode_motifs.csv"
    rules = list()
    with open(file, "r") as f:
        for l in f:
            r = l.split("=")[1].split(" \n")[0]
            label = int(l.split(" ")[3].split(":")[1])
            rules.append((label, r,l))
    return rules


def get_covers(df, rule):
    r2 = rule[1].split(" ")
    #r self.rs[18]
    #r = list(map((lambda r: r if r[:3]!="l_3" else  "a"+r[4:]), r2))
    r=r2
    rule = ps.Conjunction([ps.EqualitySelector(el, 1)for el in r])
    return df[rule.covers(df)]

import torch

def rule_converter(r):
    c = r.split(" ")
    layer = int(c[0][2])
    components = list()
    for el in c:
        components.append(int(el[5:]))
    out = layer, components
    return out

def get_egos(datas, activations, motif, eval=None):
    ids= get_covers(activations, motif)
    act_ids = list()
    acts = [0,0]
    actsg = [0,0]
    acts_df= [0, 0]
    acts_df_g= [0, 0]
    diff =0
    for g_id, (graph, features, label) in enumerate(tqdm(datas)):
        if eval is not None:
            l, comp= rule_converter(motif[1])
            embs = eval.embeddings(torch.tensor(features), torch.tensor(graph))
            cls= eval.decoder(embs[-1], None)[0].argmax().item()
            embs= embs[l]
            s = [RuleEvaluator.activate_static((l,comp),emb) for emb in embs[:int(features.sum())]]
            acts[cls]+=sum(s)
            actsg[cls]+= any(s)

        if len(ids.index[ids["id"] == g_id])==0:
            continue
        i0 = activations.index[activations["id"] == g_id][0]
        indexes = ids.index[ids["id"] == g_id].tolist()
        cls2 = int(activations["class"][i0])

        for i in indexes:
            act_ids.append((get_neigborhood(graph, features, [i-i0], int(motif[1][2])+1), (cls2, g_id, i-i0)))

        """diff+=(cls == cls2)
        acts_df_g[cls2]+=1
        acts_df[cls2]+=len(indexes)"""


    return act_ids#, acts,actsg
    #TODO




def get_negative(datas, activations, motif):
    ids= get_covers(activations, motif)
    act_ids = list()
    for g_id, (graph, features, label) in enumerate(tqdm(datas)):
        if len(ids.index[ids["id"] == g_id])==0:


            dense_adj = to_dense_adj(torch.tensor(graph)).numpy()[0]

            g = nx.from_numpy_matrix(dense_adj)  # nx.Graph()
            g.add_nodes_from([(i, {"x": el}) for i, el in enumerate(np.argmax(features, axis=1))])
            # g.add_edges_from(graph.transpose().tolist())

            act_ids.append((g, 0))

    return act_ids


def save_egos(egos, dataset, motif_id, firstline=None, dirname="ego"):

    with open("results/"+dirname+"/"+dataset+"_"+str(motif_id)+"labels_egos.txt", "w") as f:
        if firstline is not None:
            f.write("# "+firstline)
        for i, (g,l) in enumerate(egos):
            f.write("t # "+str(i)+"# "+str(l)+"\n")
            for n in g.nodes:
                label = g.nodes[n]["x"]
                f.write("v "+str(n)+" "+ str(label)+"\n")
            for a in g.adj:
                for b in g.adj[a]:
                    f.write("e "+str(a)+" "+ str(b)+" 0 \n")
        f.write("t # -1")


def to_gspan(dataset, motifs):
    graphs, features, labels, _, _, _ = load_dataset(dataset)
    datas = list(zip(graphs, features, labels))
    activation = load_activations(dataset)
    for motif_id in motifs:
        motifs = load_motifs(dataset)[motif_id]
        egos = get_egos(datas, activation, motifs)
        negative = get_negative(datas, activation,motifs)
        save_egos(egos+negative, dataset, motif_id)


def extract_egos(dataset, rules, recompute=False):
    graphs, features, labels, _, _, _ = load_dataset(dataset)
    model, checkpoint = model_selector("GNN",
                                       dataset,
                                       pretrained=True,
                                       return_checkpoint=True)
    datas = list(zip(graphs, features, labels))
    activation = load_activations(dataset)
    for motif_id in rules:

        motifs = load_motifs(dataset)[motif_id]
        egos = get_egos(datas, activation, motifs, model)

        save_egos(egos, dataset, motif_id, firstline=motifs[2],dirname="activ_ego")

"""
Mutag, motifs : 20, 40, 50
Aids : motifs 20, 30, 40 et 50
bbbp, motifs : 20, 30, 40, 50
"""


def test():
    dset= "mutag"
    model ="GNN"
    explainer= "ACT"
    pol_name = "r_list_and_label_rules2"+"activation_encode"+"ego"
    filename = "results/trace/fidelity/"+dset+"_"+model+"_"+explainer+"_"+pol_name+".csv"
    df = pd.read_csv(filename)
    return df

ba_rules =range(19)
aids_rules =range(59)#[4,14, 20,48]

bbbp_rules = range(60)# [14,39,7, 1,8,40, 3,21]
mutag_rules =  range(59)# range(100)#[3,7, 9,23,28]
dd_rules =range(66)
prot_rules=range(31)
#test()

Number_of_rules = dict([("aids", 60), ('mutag', 60), ("BBBP", 60), ("PROTEINS_full", 28), ("DD", 47), ("ba2", 19)])
"""#for dataset in ["DD", "PROTEINS_full"]:
    #for r in range(Number_of_rules[dataset]):
    extract_egos(dataset, range(Number_of_rules[dataset]))
    extract_egos(dataset, range(Number_of_rules[dataset]))"""
to_gspan("DD", range(0, Number_of_rules["DD"]))
#extract_egos("PROTEINS_full", range(Number_of_rules["PROTEINS_full"]))

"""
for r in ba_rules:
    to_gspan("ba2", r)


for r in mutag_rules:
    extract_egos("mutag",r)

for r in aids_rules:
    to_gspan("aids",r)

for r in bbbp_rules:
    to_gspan("bbbj",r)

for r in prot_rules:
    to_gspan("PROTEINS_full", r)
for r in dd_rules:
    to_gspan("DD",r)
"""
"""

from multiprocessing import Pool
from functools import partial
p = Pool(5)
#mutag»_rules = [3,7,38, 9,23,28, 33,39]
mutag_rules = [38, 9,28, 33,39]

f= partial(to_gspan,"mutag")
list(map(f,mutag_rules))

"""



def get_atom_map(dataset):
    if dataset=="aids":
        atom_map = {0 : "C", 1 : "O", 2 : "N", 3 : "Cl", 4 : "F", 5 : "S", 6 : "Se", 7 : "P", 8 : "Na", 9 : "I", 10 : "Co", 11 : "Br", 12 : "Li", 13 : "Si", 14 : "Mg", 15 : "Cu", 16 : "As", 17 : "B", 18 : "Pt", 19 : "Ru", 20 : "K", 21 : "Pd", 22 : "Au", 23 : "Te", 24 : "W", 25 : "Rh", 26 : "Zn", 27 : "Bi", 28 : "Pb", 29 : "Ge", 30 : "Sb", 31 : "Sn", 32 : "Ga", 33 : "Hg", 34 : "Ho", 35 : "Tl", 36 : "Ni", 37 : "Tb"}
    if dataset =="mutag":
        atom_map = {0 : "C", 1 : "O", 2 : "Cl", 3 : "H", 4 : "N", 5 : "F", 6 : "Br", 7 : "S", 8 : "P", 9 : "I", 10 : "Na", 11 : "K", 12 : "Li", 13 : "Ca"}
    if dataset == "BBBP":
        atom_map = ["C", "N", "O", "S", "P", "BR", "B", "F", "CL", "I", "H", "NA", "CA"]

        atom_map = dict(enumerate(atom_map))
    #atom_map.update({el+100: "Sel "+v for el,v in atom_map.items()})
    return atom_map
def ego_plots(dataset, motif ):
    atom_map = get_atom_map(dataset)
    graphs = ego_read("results/egos/"+dataset+"_"+str(motif)+"_egos.txt")
    plot_and_save(graphs,atom_map, dataset,motif,0,dir="egoplots")

def ego_read(filename):
    out = (list(), list())

    with open(filename, "r") as f:
        l = f.readline()

        for _ in range(20):
            if int(l.split(" ")[2]) == -1:
                return out


            g,l =parse_subgraph_ego(f)
            out[1].append(g)
            out[0].append(0)
            #while l!="\n":

    return out

def parse_subgraph_ego(fp):

    nodes = list()
    edges = list()
    l= fp.readline()
    while l[0]== "v":
        _, n, label = l.split(" ")
        nodes.append((int(n),{"x": int(label)}))
        l = fp.readline()

    while l[0]== "e":
        n1,n2 = l.split(" ")[1:3]
        edges.append((int(n1),int(n2)))

        l = fp.readline()

    #nodes = [(el, {"x":kwargs["atoms"][cl]}) for (el,cl),_ in lines]
    #edges = [(el[0], nb) for el,nbs in lines for nb in nbs]
    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    return g.to_undirected(),l




def plots(dataset, motif,support ):
    atom_map = get_atom_map(dataset)

    #atom_map.update({el+100: "Sel "+v for el,v in atom_map.items()})

    graphs = read_subgroup("results/egos/gspan/"+dataset+"_"+str(motif)+"_resGspan_"+str(support)+".txt")
    plot_and_save(graphs,atom_map, dataset,motif,support)

def read_subgroup(filename):
    out = (list(), list())
    with open(filename, "r") as f:
        l = f.readline()
        while True:
            while l[0]!="t":
                l = f.readline()
                if not l:
                    return out

            g =parse_subgraph(f)
            out[1].append(g)
            while l[0] !="S":
                l = f.readline()
            out[0].append(int(l.split(" ")[1]))
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

    while l!= "\n":
        _, n1,n2, _ = l.split(" ")
        edges.append((int(n1),int(n2)))

        l = fp.readline()

    #nodes = [(el, {"x":kwargs["atoms"][cl]}) for (el,cl),_ in lines]
    #edges = [(el[0], nb) for el,nbs in lines for nb in nbs]
    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    return g.to_undirected()



def plot_and_save(graphs, atom_map, dataset, motif,support ,dir="gspan/plots"):
    for i, (ng, g)  in enumerate(zip(*graphs)):
        pl = plot_mol_emp(g,atom_map)

        pl.savefig("results/egos/"+dir+"/"+dataset+"_"+str(motif)+"_"+str(support)+"_"+str(i)+"_"+str(ng)+".png")
        plt.close()
def plot_mol_emp(molecule, atom_map):
    ret=plt.figure(figsize=(4,4))
    #for i, el in enumerate(molecules):
    mol = molecule#to_networkx(molecule,node_attrs="x").to_undirected()
    labels= {node: atom_map[mol.nodes[node]["x"]]#"%1.2f"%float(mol_values[i][0][1].sum()))
             for i,node in enumerate(mol.nodes()) }
    nodes = {node for node in mol.nodes()}
    #plt.subplot(size[0],size[1], i+1)
    nx.draw_kamada_kawai(mol, node_size=1, with_labels=False)

    nx.draw_kamada_kawai(mol, nodelist=nodes, labels=labels, with_labels=True)
    #nx.draw_kamada_kawai(mol, nodelist=nodeslist, node_color="r")
    #nx.draw_kamada_kawai(mol, nodelist=centers, node_color="y")

    return ret
"""
for r in aids_rules:
    ego_plots("mutag",r)


for r in bbbp_rules:
    ego_plots("BBBP",r)

for r in mutag_rules:
    ego_plots("mutag",r)
"""

"""
for s in[25,50,100, 200]  :
    for r in aids_rules:
        plots("aids",r,s)


    for r in bbbp_rules:
        plots("BBBP",r,s)

    for r in mutag_rules:
        plots("mutag",r,s)
"""

"""
def plot_dset(dataset):
    graphs, features, _, _, _, _ = load_dataset(dataset)
    datas=list(zip(graphs, features))
    atom_map= get_atom_map(dataset)

    act_ids = list()
    for g_id, (graph, features) in enumerate(tqdm(datas)):
        dense_adj =to_dense_adj(torch.tensor(graph)).numpy()[0]
        g =nx.from_numpy_matrix(dense_adj)# nx.Graph()
        g.add_nodes_from([(i, {"x" : el}) for i, el in enumerate(np.argmax(features, axis=1)) ])
        #g.add_edges_from(graph.transpose().tolist())
        g = g.subgraph(max(nx.connected_components(g),key=len)).copy()
        pl = plot_mol_emp(g, atom_map)

        pl.savefig("results/egos/datasets/"+dataset+"_""_"+str(g_id)+".png")
        plt.close()

    return act_ids

plot_dset("aids")
plot_dset("BBBP")
plot_dset("mutag")
"""