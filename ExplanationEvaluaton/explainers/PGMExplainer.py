from math import sqrt

import torch
import torch_geometric as ptgeom
from torch import nn
from torch.optim import Adam
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from tqdm import tqdm

from ExplanationEvaluation.explainers.BaseExplainer import BaseExplainer
from ExplanationEvaluation.utils.graph import index_edge

"""
This is an adaption of the GNNExplainer of the PyTorch-Lightning library. 

The main similarity is the use of the methods _set_mask and _clear_mask to handle the mask. 
The main difference is the handling of different classification tasks. The original Geometric implementation only works for node 
classification. The implementation presented here also works for graph_classification datasets. 

link: https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/gnn_explainer.html
"""
from scipy.special import softmax
import numpy as np
class PGMExplainer(BaseExplainer):
    """
    A class encaptulating the GNNexplainer (https://arxiv.org/abs/1903.03894).
    
    :param model_to_explain: graph classification model who's predictions we wish to explain.
    :param graphs: the collections of edge_indices representing the graphs
    :param features: the collcection of features for each node in the graphs.
    :param task: str "node" or "graph"
    :param epochs: amount of epochs to train our explainer
    :param lr: learning rate used in the training of the explainer
    :param reg_coefs: reguaization coefficients used in the loss. The first item in the tuple restricts the size of the explainations, the second rescticts the entropy matrix mask.
    
    :function __set_masks__: utility; sets learnable mask on the graphs.
    :function __clear_masks__: utility; rmoves the learnable mask.
    :function _loss: calculates the loss of the explainer
    :function explain: trains the explainer to return the subgraph which explains the classification of the model-to-be-explained.
    """
    def __init__(self, model_to_explain, graphs, features, task,**kwargs):
        super().__init__(model_to_explain, graphs, features, task)
        self.top_node= 5

        self.policy_name = kwargs["policy_name"]

        self.k_top = kwargs.get("k",0)
        self.perturb_mode = kwargs.get("perturb_mode","mean")

        self.perturb_indicator = kwargs.get("perturb_indicator","diff")
    def prepare(self, args):
        """Nothing is done to prepare the GNNExplainer, this happens at every index"""
        return

    def explain(self, index):
        """
        Main method to construct the explanation for a given sample. This is done by training a mask such that the masked graph still gives
        the same prediction as the original graph using an optimization approach
        :param index: index of the node/graph that we wish to explain
        :return: explanation graph and edge weights
        """
        index = int(index)

        # Prepare model for new explanation run
        self.model_to_explain.eval()

        feats = self.features[index].detach().requires_grad_()
        graph = self.graphs[index].detach()

        pred = self.model_to_explain.forward(feats, graph)#graph, graph.ndata['feat'],graph.edata['feat'],snorm_n, snorm_e)
        #pred = self.model_to_explain(feats, graph)
        soft_pred = np.asarray(softmax(np.asarray(pred[0].data)))
        pred_threshold = 0.1*np.max(soft_pred)

        #g, f= remove_self(graph, feats)
        e = Graph_Explainer(self.model_to_explain, graph,feats,

                               perturb_feature_list = list(range(len(feats[0]))),
                               perturb_mode = self.perturb_mode,
                               perturb_indicator = self.perturb_indicator)
        pgm_nodes, p_values, candidates = e.explain(num_samples = 300, percentage = 10,
                                top_node = 5, p_threshold = 0.05, pred_threshold = pred_threshold)
        label = np.argmax(soft_pred)

        expl_graph_weights= get_expl_graph_weight_pond(pgm_nodes, graph, self.policy_name,self.k_top )

        return graph, expl_graph_weights

import torch




def get_expl_graph_weight_pond(nodes, graph,  policy_name, K=5):

    #test len_graph == len_mol
    dense_adj =to_dense_adj(graph).numpy()[0]
    mol = nx.from_numpy_matrix(dense_adj)
    adj_matrix = torch.tensor([[
            (1 if dense_adj[i, j] and (i in nodes or j in nodes) else 0) for i in range(len(mol))] for j in range(len(mol))])
    """if policy_name == "decay":
        adj_matrix = torch.tensor([[
            (pond[i]+ pond[j] if dense_adj[i, j] and i!=j else 0) for i in range(len(mol))] for j in range(len(mol))])
        adj_matrix = adj_matrix/adj_matrix.sum()
    elif policy_name[:3] == "top":
        adj_matrix = torch.tensor([[
            (pond[i]+ pond[j] if dense_adj[i, j]and i!=j else 0) for i in range(len(mol))] for j in range(len(mol))])"""


    expl_graph_weights = torch.zeros(graph.size(1))
    for pair in graph.T:  # Link explanation to original graph
        t = index_edge(graph, pair)

        expl_graph_weights[t] = (adj_matrix[pair[0],pair[1]]).clone().detach().type(expl_graph_weights.dtype)


    return expl_graph_weights

from torch_geometric.utils import to_dense_adj, dense_to_sparse


import networkx as nx





def remove_self(graph, feats):
    graph = to_dense_adj(graph)
    first0 = (graph[0]-torch.eye(graph.shape[1])).max(axis=1).values.argmin()

    return dense_to_sparse(graph[0,:first0,:first0])[0],feats[:first0,:]



from graphviz import Digraph
from torch.autograd import Variable
import torch


def make_dot(var, params=None):
    if params is not None:
        assert isinstance(params.values()[0], Variable)
        param_map = {id(v): k for k, v in params.items()}

    node_attr = dict(style="filled", shape="box", align="left", fontsize="12", ranksep="0.1", height="0.2")
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()

    def size_to_str(size):
        return "(" + (", ").join(["%d" % v for v in size]) + ")"

    def add_nodes(var):
        if var not in seen:
            if torch.is_tensor(var):
                dot.node(str(id(var)), size_to_str(var.size()), fillcolor="orange")
                dot.edge(str(id(var.grad_fn)), str(id(var)))
                var = var.grad_fn
            if hasattr(var, "variable"):
                u = var.variable
                name = param_map[id(u)] if params is not None else ""
                node_name = "%s\n %s" % (name, size_to_str(u.size()))
                dot.node(str(id(var)), node_name, fillcolor="lightblue")
            else:
                dot.node(str(id(var)), str(type(var).__name__))
            seen.add(var)
            if hasattr(var, "next_functions"):
                for u in var.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(var)))
                        add_nodes(u[0])
            if hasattr(var, "saved_tensors"):
                for t in var.saved_tensors:
                    dot.edge(str(id(t)), str(id(var)))
                    add_nodes(t)

    add_nodes(var)
    return dot


import time

import pandas as pd
from scipy.special import softmax
from pgmpy.estimators import PC as ConstraintBasedEstimator
from pgmpy.estimators.CITests import chi_square



def n_hops_A(A, n_hops):
    # Compute the n-hops adjacency matrix
    adj = torch.tensor(A, dtype=torch.float)
    hop_adj = power_adj = adj
    for i in range(n_hops - 1):
        power_adj = power_adj @ adj
        prev_hop_adj = hop_adj
        hop_adj = hop_adj + power_adj
        hop_adj = (hop_adj > 0).float()
    return hop_adj.numpy().astype(int)


class Graph_Explainer:
    def __init__(
            self,
            model,
            graph,
            X_feat,
            num_layers=None,
            perturb_feature_list=None,
            perturb_mode="mean",  # mean, zero, max or uniform
            perturb_indicator="diff",  # diff or abs
            print_result=1,
            snorm_n=None,
            snorm_e=None
    ):
        self.model = model
        self.model.eval()
        self.graph = graph


        self.num_layers = num_layers
        self.perturb_feature_list = perturb_feature_list
        self.perturb_mode = perturb_mode
        self.perturb_indicator = perturb_indicator
        self.print_result = print_result
        self.X_feat = X_feat
        #self.E_feat = graph.edata['feat'].numpy()

    def perturb_features_on_node(self, feature_matrix, node_idx, random=0):

        #X_perturb = feature_matrix.clone()
        #perturb_array = torch.zeros(feature_matrix.shape[1])#feature_matrix[node_idx].clone()
        #epsilon = 0.05 * self.X_feat.detach().max(0).values
        seed = np.random.randint(2)

        if random == 1:
            if seed == 1:
                #for i in range(perturb_array.shape[0]):
                #if i in self.perturb_feature_list:
                if self.perturb_mode == "mean":
                    #perturb_array[i] = feature_matrix[:, i].detach().mean()#np.mean(feature_matrix[:, i].detach().mean()
                    feature_matrix[node_idx,self.perturb_feature_list] = feature_matrix[:, self.perturb_feature_list].mean(axis=0)
                elif self.perturb_mode == "zero":
                    #perturb_array[i] = 0
                    feature_matrix[node_idx,self.perturb_feature_list] =0
                elif self.perturb_mode == "max":
                    #perturb_array[i] = feature_matrix[:, i].detach().max()
                    feature_matrix[node_idx,self.perturb_feature_list] = feature_matrix[:, :].max(axis=0)
                elif self.perturb_mode == "uniform":
                    return -1
                    """ #perturb_array[i] = feature_matrix[node_idx][i].detach() + np.random.uniform(low=-epsilon[i], high=epsilon[i])
                    feature_matrix[node_idx]
                    if feature_matrix[node_idx][i] < 0:
                        perturb_array[i] = 0
                    elif feature_matrix[node_idx][i] > np.max(self.X_feat, axis=0)[i]:
                        perturb_array[i] = np.max(self.X_feat, axis=0)[i]"""

                #feature_matrix[node_idx] = perturb_array

        return feature_matrix

    def batch_perturb_features_on_node(self, num_samples, index_to_perturb,
                                       percentage, p_threshold, pred_threshold):
        X_torch = torch.tensor(self.X_feat, dtype=torch.float)
        #E_torch = torch.tensor(self.E_feat, dtype=torch.float)
        pred_torch = self.model.forward(X_torch, self.graph)#, E_torch)
        soft_pred = np.asarray(softmax(np.asarray(pred_torch[0].data)))
        pred_label = np.argmax(soft_pred)
        num_nodes = self.X_feat.shape[0]
        Samples = []
        for iteration in range(num_samples):
            X_perturb = self.X_feat.detach().clone()
            sample = []
            for node in range(num_nodes):
                if node in index_to_perturb:
                    seed = np.random.randint(100)
                    if seed < percentage:
                        latent = 1
                        X_perturb = self.perturb_features_on_node(X_perturb, node, random=latent)
                    else:
                        latent = 0
                else:
                    latent = 0
                sample.append(latent)

            X_perturb_torch = torch.tensor(X_perturb, dtype=torch.float)
            pred_perturb_torch = self.model.forward(X_perturb_torch,self.graph )#, E_torch)
            soft_pred_perturb = np.asarray(softmax(np.asarray(pred_perturb_torch[0].data)))

            pred_change = np.max(soft_pred) - soft_pred_perturb[pred_label]

            sample.append(pred_change)
            Samples.append(sample)

        Samples = np.asarray(Samples)
        if self.perturb_indicator == "abs":
            Samples = np.abs(Samples)

        top = int(num_samples / 8)
        top_idx = np.argsort(Samples[:, num_nodes])[-top:]
        for i in range(num_samples):
            if i in top_idx:
                Samples[i, num_nodes] = 1
            else:
                Samples[i, num_nodes] = 0

        return Samples

    def explain(self, num_samples=10, percentage=50, top_node=None, p_threshold=0.05, pred_threshold=0.1):

        num_nodes = self.X_feat.shape[0]
        if top_node == None:
            top_node = int(num_nodes / 20)

        #         Round 1

        #node_explore = range((self.X_feat.detach().numpy().sum(axis=1)+[0]).argmin()-1)
        node_explore= range((np.concatenate([self.X_feat.detach().numpy().sum(axis=1), [0]])).argmin()-1)
        Samples = self.batch_perturb_features_on_node(int(num_samples / 2), node_explore, percentage,
                                                      p_threshold, pred_threshold)

        data = pd.DataFrame(Samples)
        est = ConstraintBasedEstimator(data)

        p_values = []
        candidate_nodes = []

        target = num_nodes  # The entry for the graph classification data is at "num_nodes"
        for node in range(num_nodes):
            chi2, _,p = chi_square(node, target, [], data, boolean=False)
            p_values.append(p)

        number_candidates = int(top_node * 4)
        candidate_nodes = np.argpartition(p_values, number_candidates)[0:number_candidates]

        #         Round 2
        Samples = self.batch_perturb_features_on_node(num_samples, candidate_nodes, percentage,
                                                      p_threshold, pred_threshold)
        data = pd.DataFrame(Samples)
        est = ConstraintBasedEstimator(data)

        p_values = []
        dependent_nodes = []

        target = num_nodes
        for node in node_explore:
            chi2, _,p = chi_square(node, target, [], data,boolean=False)
            p_values.append(p)
            if p < p_threshold:
                dependent_nodes.append(node)

        top_p = np.min((top_node, num_nodes - 1,len(p_values)-1))
        ind_top_p = np.argpartition(p_values, top_p)[0:top_p]
        pgm_nodes = list(ind_top_p)

        return pgm_nodes, p_values, candidate_nodes