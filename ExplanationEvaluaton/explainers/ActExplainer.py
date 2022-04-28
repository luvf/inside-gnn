 »from math import sqrt
from scipy.special import softmax
from tqdm import tqdm
from torch_geometric.nn import MessagePassing
import os
from functools import partial
import torch
import torch_geometric as ptgeom
from torch import nn
from torch.optim import Adam
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from tqdm import tqdm
from torch_geometric.utils import to_dense_adj, dense_to_sparse
import numpy as np
import pandas as pd
import pysubgroup as ps
from torch_geometric.utils.convert import to_networkx
import networkx as nx
from itertools import accumulate

from ExplanationEvaluation.explainers.BaseExplainer import BaseExplainer
from ExplanationEvaluation.utils.graph import index_edge
from ExplanationEvaluation.datasets.gspan_parser import gspan_parser

from numba import jit
from multiprocessing import Pool

import functools


class ActExplainer(BaseExplainer):
    """

    """
    def __init__(self, model_to_explain, graphs, features, task, config,labels,**kwargs):
        super().__init__(model_to_explain, graphs, features, task)
        self.config=config
        self.labels = labels
        #self.rules = ("l_0c_2 l_0c_4 l_0c_10 l_0c_18 l_0c_6 l_0c_0", "l_1c_10 l_1c_13 l_1c_15 l_1c_11 l_1c_4 l_1c_19 l_1c_5 l_1c_0 l_1c_8 l_1c_6 l_1c_18 l_1c_16 l_1c_12 l_1c_3", "l_1c_15 l_1c_14", "l_1c_16 l_1c_13 l_1c_15 l_1c_11 l_1c_4 l_1c_19 l_1c_5 l_1c_0 l_1c_8 l_1c_6 l_1c_18", "l_1c_13 l_1c_14", "l_2c_3 l_2c_9 l_2c_12 l_2c_5 l_2c_1 l_2c_4 l_2c_18 l_2c_10 l_2c_13 l_2c_6 l_2c_7 l_2c_15 l_2c_14", "l_2c_2 l_2c_0 l_2c_5 l_2c_11 l_2c_17 l_2c_1", "l_2c_3 l_2c_9 l_2c_7 l_2c_15 l_2c_6 l_2c_10 l_2c_19 l_2c_0 l_2c_11 l_2c_17", "l_2c_3 l_2c_9 l_2c_7 l_2c_15 l_2c_16 l_2c_0 l_2c_11 l_2c_17 l_2c_19 l_2c_8", "l_2c_16 l_2c_0 l_2c_11 l_2c_17 l_2c_19 l_2c_8 l_2c_18 l_2c_4 l_2c_5 l_2c_1 l_2c_2", "l_2c_3 l_2c_9 l_2c_7 l_2c_15 l_2c_6 l_2c_10 l_2c_16 l_2c_0 l_2c_11 l_2c_17 l_2c_19 l_2c_18 l_2c_5", "l_2c_8 l_2c_0 l_2c_11 l_2c_17 l_2c_19 l_2c_18 l_2c_2 l_2c_5 l_2c_1", "l_2c_13 l_2c_5 l_2c_1 l_2c_10 l_2c_18 l_2c_19 l_2c_0 l_2c_11 l_2c_17 l_2c_2 l_2c_4", "l_2c_8 l_2c_0 l_2c_11 l_2c_17 l_2c_19 l_2c_2 l_2c_5 l_2c_1", "l_2c_8 l_2c_0 l_2c_11 l_2c_17 l_2c_19 l_2c_10 l_2c_18 l_2c_4 l_2c_5 l_2c_1 l_2c_2", "l_2c_13 l_2c_5 l_2c_1 l_2c_10 l_2c_8 l_2c_0 l_2c_11 l_2c_17 l_2c_2 l_2c_19 l_2c_18 l_2c_14 l_2c_4")
        self.reset_storage=False
        self.policy_name = kwargs["policy_name"]

        self.k_top = kwargs.get("k",0)


        self.motif = kwargs["motifs"]




    def prepare(self, indices):
        """Nothing is done to prepare the GNNExplainer, this happens at every index"""



        #df = self._embedding_to_df(embeddings, outputs)
        #self.df= df
        filename = 'results/trace/'+self.config.dataset+"_"+self.config.model+"_"+self.config.explainer+"_activation.csv"
        self.df = self._get_df(filename, partial(self._embedding_to_df))
        #self._save_df(df,"_activation")
        #self.test_graphs()
        self.rules = labeled_rules(self.config.dataset,self.motif)
        self.policies, pol_names = get_policies(self.rules, self.policy_name,self.motif)
        self.indices = indices

        #self.policy = "test1"
        filename = "results/trace/fidelity/"+self.config.dataset+"_"+self.config.model+"_"+self.config.explainer+"_"+pol_names+".csv"

        self.covers = get_covers(self.df, self.rules)

        self.reset_storage = False
        self.perturbdf = self._get_df(filename,self.fidelity_nodes)
        """for anode in [not self.and_nodes, self.and_nodes]:
            anode= "and_" if anode else ""
            filename = "results/trace/fidelity/"+self.config.dataset+"_"+self.config.model+"_"+self.config.explainer+"_"+anode+".csv"
        """
        return None

    def _call_subgroup(self):
        pass

    def _get_df(self, file, function):

        if self.reset_storage or not os.path.exists(file):
            df = function()
            df.to_csv(file, compression="infer",float_format='%.3f')
        df =pd.read_csv(file)
        return df


    def _save_df(self,dataframe,name, dir_path="trace"):
        filename = self.config.dataset+"_"+self.config.model+"_"+self.config.explainer+name

        name = 'results/'+dir_path+'/'+filename+"csv"
        dataframe.to_csv(name+".csv", compression="infer",float_format='%.3f')
        #pd.Series(dset.indices).to_csv(name+ "indices.csv", compression="infer")



    def _embedding_to_df(self, labels =True):
        #layers = [20, 20, 20]
        #acc = [0]+ list(accumulate(layers))
        #split = [embeddings[:,acc[i]: acc[i+1]] for i in range(len(layers))]
        if len(self.features.shape) == 2:
            embeddings = list(map(lambda x : x.detach().numpy(),self.model_to_explain.embeddings(self.features,self.graphs)))
            outputs = self.model_to_explain(self.features, self.graphs).detach().numpy()
        elif len(self.features.shape) == 3:
            embeddings = [list(map(lambda x : x.detach().numpy(),self.model_to_explain.embeddings(f,g))) for f, g in zip(self.features, self.graphs)]
            embeddings = np.array(embeddings)

            outputs = np.array([self.model_to_explain(f,g).detach().numpy() for f, g in zip(self.features, self.graphs)]).squeeze()
        else :
            raise NotImplementedError("Unknown graph data type")

        atoms = self.features
        embeddings= np.array(embeddings)
        if len(embeddings.shape) == 3 :
            embeddings= np.transpose(embeddings, [1,0,2])
            graph_sizes=1
        elif len(embeddings.shape) == 4:
            embeddings = np.transpose(embeddings, [0,2,1,3])
            shape = embeddings.shape
            graph_sizes = shape[1] ##assume all graphs have same size ok in this framework
            embeddings = embeddings.reshape((shape[0]*shape[1], shape[2],shape[3]))
            atoms = atoms.reshape((shape[0]*shape[1],-1))

        layersize = embeddings.shape[-1]
        nlayer = embeddings.shape[-2]


        d2 = np.array([[el//graph_sizes]+ [x for l in range(nlayer) for x in embeddings[el,l]] for el in range(len(embeddings)) ])

        dd = pd.DataFrame(d2, columns=["id"]+["l_"+ str(i//layersize) +"c_"+ str(i%layersize) for i in range(nlayer*layersize)])

        inputs = pd.DataFrame(atoms.numpy(), columns=["a_"+str(i) for i in range(len(atoms[0]))])

        dd = pd.concat([dd,inputs], axis=1)

        #labels = pd.DataFrame({"class_"+str(i): [targets[i].y.numpy()[i] for el in data[0] ] for i in range(len(targets[0]))})
        #preds = pd.DataFrame({"pred_"+str(i): [outputs[el//graph_sizes][i] for el in range(len(embeddings)) ] for i in range(len(outputs[0]))})
        #pred_base = pd.DataFrame({"pred_base" : [np.argmax(outputs[el//graph_sizes]) for el in range(len(embeddings)) ]})
        pred_bin = pd.DataFrame({"class" : [np.argmax(outputs[el//graph_sizes])>0 for el in range(len(embeddings)) ]})
        if labels:
            if len(self.labels.shape)==1:
                pred_true = pd.DataFrame({"true_class" : [self.labels[el//graph_sizes].item()>0 for el in range(len(embeddings)) ]})
            else :
                pred_true = pd.DataFrame(
                    {"true_class": [np.argmax(self.labels[el // graph_sizes]).item() > 0 for el in range(len(embeddings))]})
        dd = pd.concat([dd, pred_true,pred_bin], axis=1)
        dd.update(dd.filter(regex="l_[0-9]*c_[0-9]*")>0)
        return dd



    def explain(self, index):
        """
        Main method to construct the explanation for a given sample. This is done by training a mask such that the masked graph still gives
        the same prediction as the original graph using an optimization approach
        :param index: index of the node/graph that we wish to explain
        :return: explanation graph and edge weights
        """

        if self.type == 'graph': # c'est le plus simple«0
            return explain_graph_with_motifs(self.perturbdf, self.graphs[index], index, self.covers,self.rules, self.policies,self.policy_name,self.k_top)

    #obsolete
    def test_graphs(self):
        k = 20
        actviation = list(range(16))+[20]
        for k in actviation:
            file = "ExplanationEvaluation/datasets/activations/ba2_activation"+str(k)+".txt"
            graphs, features = list(zip(*gspan_parser(file)))

            if len(self.features.shape) == 2:
                embeddings = list(map(lambda x : x.detach().numpy(),self.model_to_explain.embeddings(features,graphs)))
                outputs = self.model_to_explain(self.features, self.graphs).detach().numpy()
            elif len(self.features.shape) == 3:
                embeddings = [list(map(lambda x : x.detach().numpy(),self.model_to_explain.embeddings(f,g[0]))) for f, g in zip(features, graphs)]
                outputs = np.array([self.model_to_explain(f,g[0]).detach().numpy() for f, g in zip(features, graphs)]).squeeze()
                embeddings = [[i]+[x for l in e for x in l[a]]+[np.argmax(outputs[i])>0] for i,e in enumerate(embeddings) for a in range(len(e[0]))]
            else :
                raise NotImplementedError("Unknown graph data type")
            layersize =20
            nlayer=3
            dd = pd.DataFrame(embeddings, columns = ["id"]+["l_"+ str(i//layersize) +"c_"+ str(i%layersize) for i in range(nlayer*layersize)]+["class"])
            inputs = pd.DataFrame(np.concatenate(features), columns=["a_"+str(i) for i in range(10)])
            dd = pd.concat([dd,inputs], axis=1)
            self._save_df(dd, name= "_gspan_rerun_"+str(k))


    def fidelity_nodes(self):
        perturbation_array = np.zeros((len(self.graphs),len(self.policies)+1))
        if self.type == 'graph': # c'est le plus
            """fun = partial(fidelity_helper, self.df,self.rules, self.model_to_explain, self.policies,list(zip(self.graphs, self.features)))
            p = Pool(20)
            out = p.map(fun, tqdm(range(len(self.graphs))))
            return pd.DataFrame([[i]+s.tolist() for i, s in enumerate(out)], columns = ["id", "base"]+ ["r"+str(i) for i in range(len(self.policies))])


            """
            if False:
                indices = range(self.graphs)
            else :
                indices = self.indices
            for i in tqdm(indices):
                #for i, graph in enumerate(tqdm(self.graphs)):
                graph = self.graphs[i]

                expl_graph_weights = [get_expl_graph_weight_pond(self.covers, graph, i, self.rules, policy= policy,policy_name=self.policy_name,K=self.k_top) for policy in self.policies]

                base = get_score(self.model_to_explain, self.features[i], graph, torch.zeros(graph.size(1)))
                perturbed = [get_score(self.model_to_explain, self.features[i], graph,w)if w.sum()>0 else base for w in expl_graph_weights]


                perturbation_array[i,:]= [base]+ perturbed
            return pd.DataFrame([[i]+s.tolist() for i, s in enumerate(perturbation_array)], columns = ["id", "base"]+ ["r"+str(i) for i in range(len(self.policies))])

        return None


def get_score(model, f, graph ,explanation):
    score = model(f, graph, edge_weights=1.- explanation).detach().numpy()
    return softmax(score)[0,1]


def fidelity_helper(df, rules, model,policies, d, i, policy_name="node"):
    g = d[i][0]
    f = d[i][1]
    #expl_graph_weights = p.map(fun, self.policies)
    covers = get_covers(df, rules)

    expl_graph_weights = [get_expl_graph_weight_pond(covers, g, i,rules, policy= policy,policy_name= policy_name) for policy in policies]

    perturbed = [get_score(model, f, (g, w)) for w in expl_graph_weights]
    base = get_score(model, f, (graph, torch.zeros(graph.size(1))))

    return [base] + perturbed

def get_expl_graph_weight_pond(covers, graph, g_index, rules, policy, policy_name,K=5):
    pond = policy(covers, g_index, graph,rules)
    if pond.sum()==0:
        return torch.zeros(graph.size(1))

    #test len_graph == len_mol
    dense_adj =to_dense_adj(graph).numpy()[0]
    mol = nx.from_numpy_matrix(dense_adj)
    if policy_name == "node":
        adj_matrix = torch.tensor([[
            (max(pond[i],pond[j])>=1 if dense_adj[i,j] else 0) for i in range(len(mol))] for j in range(len(mol))])
    elif policy_name == "ego":
        adj_matrix = torch.tensor([[
            (max(pond[i], pond[j]) >= 1 if dense_adj[i, j] else 0) for i in range(len(mol))] for j in range(len(mol))])
    elif policy_name == "decay":
        adj_matrix = torch.tensor([[
            (pond[i]+ pond[j] if dense_adj[i, j] else 0) for i in range(len(mol))] for j in range(len(mol))])
    elif policy_name[:3] == "top":
        adj_matrix = torch.tensor([[
            (pond[i]+ pond[j] if dense_adj[i, j] else 0) for i in range(len(mol))] for j in range(len(mol))])



    expl_graph_weights = torch.zeros(graph.size(1))
    for pair in graph.T:  # Link explanation to original graph
        t = index_edge(graph, pair)
        if policy_name[:3] != "top":
            expl_graph_weights[t] = (adj_matrix[pair[0],pair[1]]>0.9).clone().detach().type(expl_graph_weights.dtype)
        else:
            expl_graph_weights[t] = (adj_matrix[pair[0],pair[1]]).clone().detach().type(expl_graph_weights.dtype)
    if policy_name[:3] == "top":
        top_indices = np.argsort(expl_graph_weights)[-K:]
        top_indices = top_indices[expl_graph_weights[top_indices]>0]
        expl_graph_weights=torch.zeros(expl_graph_weights.shape)
        expl_graph_weights[top_indices]=1
        #expl_graph_weights = expl_graph_weights2.type(torch.float32)
    return expl_graph_weights

def single_rule_policy(covers, g_index, graph, rules, rule_number=0, policy="node"):
    #rule = rules[rule_number][1]
    index_zero = get_indexes(covers, g_index, rule_number)
    if len(index_zero)==0:
        return np.zeros(1)

    layer = int(rules[rule_number][1][2])+1
    if layer == 4:
        layer = 0

    dense_adj =to_dense_adj(graph).numpy()[0]
    mol = nx.from_numpy_matrix(dense_adj)
    pond = np.zeros(len(mol))

    for i in index_zero:
        dists= dict(nx.algorithms.single_target_shortest_path_length(mol,i))
        for k, v in dists.items():
            if (v <=layer):
                if policy == "node":
                    pond[int(k)] += (v <=0)
                elif policy == "ego":
                    pond[int(k)] += (v <=layer) #"1s2",1s2_top
                elif policy == "decay":
                    pond[int(k)] += 1 / (2 ** (1 + v))  # "1s2",1s2_top
                elif policy[:3]== "top":
                    pond[int(k)] += 1 / (2 ** (1 + v))  # "1s2",1s2_top
                #pond[int(k)] += (v <=layer) #"(v <=layer)"
                #pond[int(k)] += (v <=0)

    #"1s2_top"
    #return np.array([1 if i in np.argsort(pond)[-int(len(pond)*0.15):] else 0 for i in range(len(pond))])
    return pond

def get_policies(rules, policy_name, motif):
    out =list()
    #get the 0 and labeled rules
    index = [list(), list()]
    for i in range(len(rules)):
        index[rules[i][0]].append(i)
        out+=[partial(single_rule_policy, rule_number= i, policy=policy_name)]
    #out+=[partial(rule_list_policy, rule_numbers= index[0])]
    #out+=[partial(rule_list_policy, rule_numbers= index[1])]

    return out, "r_list_and_label_rules2"+get_motifs_file(motif)+policy_name


def get_motifs_file(motif):
    if motif =="base" :
        return "activation_encode"
    if motif == "selection":
        return "selection"



def labeled_rules(dataset,motif):
    names= {"ba2": ("ba2"),
            "aids" : ("Aids"),
            "BBBP": ("Bbbp"),
            "mutag": ("Mutag")}
    name = names[dataset]
    file = "ExplanationEvaluation/datasets/activations/"+ name+"/"+name+"_"+get_motifs_file(motif)+"_motifs.csv"
    rules = list()
    with open(file, "r") as f:
        for l in f:
            r = l.split("=")[1].split(" \n")[0]
            label = int(l.split(" ")[3].split(":")[1])
            rules.append((label, r))
    return rules

def get_rules(dataset):
    names= {"ba2": ("ba2"),
                 "aids" : ("Aids")}
    name = names[dataset]
    file = "ExplanationEvaluation/datasets/activations/"+ name+"/"+name+"_activation_encode_motifs.csv"
    rules = list()
    with open(file, "r") as f:
        for l in f:
            rules.append(l.split("=")[1].split(" \n")[0])
    return rules


def get_covers(df, rules):
    ids=dict()
    for i, (cl, rulet) in enumerate(rules):
        r2 = rulet.split(" ")
        #r =self.rs[18]
        r = list(map((lambda r: r if r[:3]!="l_3" else  "a"+r[4:]), r2))

        rule = ps.Conjunction([ps.EqualitySelector(el, 1)for el in r])
        ids[i] = df[rule.covers(df)]
    return ids, df

def get_indexes(covers, g_id, rule_id):
    cv, dframe = covers
    ids= cv[rule_id]
    if len(ids.index[ids["id"] == g_id])==0:
        return []
    i0 = dframe.index[dframe["id"] == g_id][0]
    indexes = ids.index[ids["id"] == g_id].tolist()
    return [int(i-i0) for i in indexes]





def explain_graph_with_motifs(perturb_df, graph, index, covers,rules, policies,policy_name="node",K=5):

    row = perturb_df[perturb_df["id"]==index].values.tolist()[0][2:-2]
    base, pol =row[0], row[1:]
    has_rule = any([(base>0.5 )!= (p>0.5) for p in pol])
    if has_rule:
        popol = np.argmax([np.abs(base-p) for p in pol])
        return graph, get_expl_graph_weight_pond(covers, graph, index, rules, policy=policies[popol], policy_name=policy_name, K=K)
    else:
        return graph,torch.zeros(graph.size(1))


    """if has_rule:
    pert = [np.abs(base-p) for p in pol]
        return graph, get_expl_graph_weight_pond(df, graph, index, rules,policy=policies[np.argmax(pert)])
    else :
        return graph,torch.zeros(graph.size(1))"""


def get_strat_neigborhood(mol, nodes, radius):
    nn = nodes.copy()
    out = [nodes.copy()]
    for i in range(radius):
        last= [el for n in nn for el in list(mol.adj[n]) ]
        last = set(last)-set(nn)
        out.append(last)
        nn = list(set(nn).union(last))
        #nbh = nx.Graph(mol.subgraph(list(g1.adj[nodes])))
    return out



def get_neigborhood(mol, nodes, radius):
    #mol = nx.Graph()
    #mol = from_numpy_matrix(molecule.numpy())
    #mol = nx.from_numpy_matrix(to_dense_adj(molecule).numpy()[0])
    #mol = to_networkx(molecule, node_attrs="x").to_undirected()
    nn = nodes.copy()
    for i in range(radius):
        nn += [el for n in nn for el in list(mol.adj[n]) ]
        nn = list(set(nn))
    #nbh = nx.Graph(mol.subgraph(list(g1.adj[nodes])))
    return nn

#obsolete
def rule_list_policy(covers, g_index, graph,rule_numbers=[]):
    #rules = [rules[nb][1] for nb in rule_numbers]

    index_zero = np.concatenate([get_indexes(covers, g_index, rule_id) for rule_id in rule_numbers])
    if len(index_zero) == 0:
        return np.zeros(1)

    dense_adj = to_dense_adj(graph).numpy()[0]
    mol = nx.from_numpy_matrix(dense_adj)
    pond = np.zeros(len(mol))

    for i in index_zero:
        dists = dict(nx.algorithms.single_target_shortest_path_length(mol, i))
        for k, v in dists.items():
            pond[int(k)] += (v <=2)  # 1/(2*(1+v))

    return pond
    # indices = np.argsort(pond)[-int(len(pond)*0.1):]
    # return np.array([1 if i in np.argsort(pond)[-int(len(pond)*0.1):] else 0 for i in range(len(pond))])
    # return pond
