import networkx as nx
import torch
import torch.nn as nn

import numpy as np
import copy
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import defaultdict
from .policy_nets import PolicyNN
from torch_geometric.utils import dense_to_sparse
import networkx
from tqdm import tqdm
from torch import as_tensor
from torch.nn.functional import one_hot, softmax
from torch_geometric.utils import dense_to_sparse
from ExplanationEvaluation.explainers.utils import RuleEvaluator, get_atoms

class RandomExplainer_subgraph():
    def __init__(self, model_to_explain, dset,dset_name,  max_node=6, max_step=10,  random=False,target_rule=None,target_metric="sum"):
        #print('Start training pipeline')
        self.gnnNets = model_to_explain
        self.dset = dset
        self.graphs= [torch.tensor(el) for el in dset[0]]
        self.features = torch.tensor(dset[1])

        self.graph= nx.Graph()
        #$     self.mol = Chem.RWMol()  ### keep a mol obj, to check if the graph is valid

        self.dset_name = dset_name

        self.rule_evaluator = RuleEvaluator(self.gnnNets, dset_name, target_rule, ["sum", "lin", "entropy", "cheb", "likelyhood", "hamming"])
        self.target_rule = self.rule_evaluator.target_rule

        self.best_score = [0]
        self.step_score = [0]
        self.size_best = [0]
        self.target_metric = target_metric
        self.rollout_graphs = list()


    def prepare(self):
        if len(self.features.shape) == 2:
            embeddings = list(map(lambda x : x.detach().numpy(),self.gnnNets.embeddings(self.features,self.graphs)))
            outputs = self.gnnNets(self.features, self.graphs).detach().numpy()
        elif len(self.features.shape) == 3:
            embeddings = [list(map(lambda x : x.detach().numpy(),self.gnnNets.embeddings(f,g))) for f, g in zip(self.features, self.graphs)]
            embeddings = np.array(embeddings)

            outputs = np.array([self.gnnNets(f,g).detach().numpy() for f, g in zip(self.features, self.graphs)]).squeeze()
        else :
            raise NotImplementedError("Unknown graph data type")

        x=1


def get_activation_score(model, dataset, dataset_name, target_rule, metrics):
    embs= get_embeddings(model,dataset)
    rule_evaluator = RuleEvaluator(model, dataset_name, target_rule, metrics=metrics,atom_labels=False)
    activated=list()
    unactivated = list()
    for i, embg in enumerate(embs[:,target_rule//20,:,:]):
        len =np.argmin(np.abs(dataset[0][1][0]-dataset[0][1][1]))
        for emb in embg[:len] :
            if rule_evaluator.activate(emb):
                activated.append(rule_evaluator.compute_score_emb(torch.tensor(emb)))
            else :
                unactivated.append(rule_evaluator.compute_score_emb(torch.tensor(emb)))

    return activated,unactivated

def get_embeddings(model, dset):
    graphs = [torch.tensor(el) for el in dset[0]]
    features = torch.tensor(dset[1])
    if len(features.shape) == 2:
        embeddings = list(map(lambda x: x.detach().numpy(), model.embeddings(features, graphs)))
        outputs = model(features, graphs).detach().numpy()
    elif len(features.shape) == 3:
        embeddings = [list(map(lambda x: x.detach().numpy(), model.embeddings(f, g))) for f, g in
                      zip(features, graphs)]
        embeddings = np.array(embeddings)

        outputs = np.array([model(f, g).detach().numpy() for f, g in zip(features, graphs)]).squeeze()
    else:
        raise NotImplementedError("Unknown graph data type")
    return embeddings