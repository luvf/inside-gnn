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
class GradExplainer(BaseExplainer):
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
        label = np.argmax(soft_pred)
        #g = make_dot(pred)
        #g.view()

        grad = torch.autograd.grad(pred[0,label], feats)[0]

        """ for j in range(features.shape[0]):
            phi0[j] = grad[j,0]"""
        #phi0 = grad[:feats.shape[0],0]
        phi0 = np.abs(grad[:feats.shape[0],:]).sum(axis=1)
        score = phi0.data.numpy()
        grad_nodes = list(np.argpartition(score, -self.top_node)[-self.top_node:])
        expl_graph_weights= get_expl_graph_weight_pond(phi0, graph, self.policy_name,self.k_top )

        return  graph, expl_graph_weights

import torch


from torch_geometric.utils import to_dense_adj, dense_to_sparse


import networkx as nx


def get_expl_graph_weight_pond(pond, graph,  policy_name, K=5):



    #test len_graph == len_mol
    dense_adj =to_dense_adj(graph).numpy()[0]
    mol = nx.from_numpy_matrix(dense_adj)

    if policy_name == "decay":
        adj_matrix = torch.tensor([[
            (pond[i]+ pond[j] if dense_adj[i, j] and i!=j else 0) for i in range(len(mol))] for j in range(len(mol))])
        adj_matrix = adj_matrix/adj_matrix.sum()
    elif policy_name[:3] == "top":
        adj_matrix = torch.tensor([[
            (pond[i]+ pond[j] if dense_adj[i, j]and i!=j else 0) for i in range(len(mol))] for j in range(len(mol))])


    expl_graph_weights = torch.zeros(graph.size(1))
    for pair in graph.T:  # Link explanation to original graph
        t = index_edge(graph, pair)

        expl_graph_weights[t] = (adj_matrix[pair[0],pair[1]]).clone().detach().type(expl_graph_weights.dtype)
    if policy_name[:3] == "top":
        top_indices = np.argsort(expl_graph_weights)[-K:]
        top_indices = top_indices[expl_graph_weights[top_indices]>0]
        expl_graph_weights=torch.zeros(expl_graph_weights.shape)
        expl_graph_weights[top_indices]=1
        #expl_graph_weights = expl_graph_weights2.type(torch.float32)
    return expl_graph_weights


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
