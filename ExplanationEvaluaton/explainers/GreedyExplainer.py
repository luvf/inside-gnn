# Import packages
import random
import time
from copy import deepcopy
from itertools import combinations

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.special
import torch
import torch_geometric
from tqdm import tqdm
from sklearn.linear_model import (LassoLars, Lasso,
                                  LinearRegression, Ridge)
from sklearn.metrics import r2_score
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.nn import GNNExplainer as GNNE
from torch_geometric.nn import MessagePassing

from src.models import LinearRegressionModel
from src.plots import (denoise_graph, k_hop_subgraph, log_graph,
                       visualize_subgraph, custom_to_networkx)





class Greedy:

    def __init__(self, data, model, gpu=False):
        self.model = model
        self.data = data
        self.neighbours = None
        self.gpu = gpu
        self.M = self.data.num_features
        self.F = self.M

        self.model.eval()

    def explain(self, node_index=0, hops=2, num_samples=0, info=False, multiclass=False, *unused):
        """
        Greedy explainer - only considers node features for explanations
        Computes the prediction proba with and without the targeted feature (repeat for all feat)
        This feature's importance is set as the normalised absolute difference in predictions above
        :param num_samples, info: useless here (simply to match GraphSVX structure)
        """
        # Store indexes of these non zero feature values
        feat_idx = torch.arange(self.F)

        # Compute predictions
        if self.gpu:
            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
            self.model = self.model.to(device)
            with torch.no_grad():
                probas = self.model(x=self.data.x.cuda(), edge_index=self.data.edge_index.cuda()).exp()[
                    node_index]
        else:
            with torch.no_grad():
                probas = self.model(x=self.data.x, edge_index=self.data.edge_index).exp()[
                    node_index]
        probas, label = torch.topk(probas, k=1)

        # Init explanations vector

        if multiclass:
            coefs = np.zeros([self.M, self.data.num_classes])  # (m, #feats)

            # Loop on all features - consider all classes
            for i, idx in enumerate(feat_idx):
                idx = idx.item()
                x_ = deepcopy(self.data.x)
                x_[:, idx] = 0.0  # set feat of interest to 0
                if self.gpu:
                    with torch.no_grad():
                        probas_ = self.model(x=x_.cuda(), edge_index=self.data.edge_index.cuda()).exp()[
                            node_index]  # [label].item()
                else:
                    with torch.no_grad():
                        probas_ = self.model(x=x_, edge_index=self.data.edge_index).exp()[
                            node_index]  # [label].item()
                # Compute explanations with the following formula
                coefs[i] = (torch.abs(probas-probas_)/probas).detach().numpy()
        else:
            #probas = probas[label.item()]
            coefs = np.zeros([self.M])  # (m, #feats)
            # Loop on all features - consider all classes
            for i, idx in enumerate(feat_idx):
                idx = idx.item()
                x_ = deepcopy(self.data.x)
                x_[:, idx] = 0.0  # set feat of interest to 0
                if self.gpu:
                    with torch.no_grad():
                        probas_ = self.model(x=x_.cuda(), edge_index=self.data.edge_index.cuda()).exp()[
                            node_index, label.item()]  # [label].item()
                else:
                    with torch.no_grad():
                        probas_ = self.model(x=x_, edge_index=self.data.edge_index).exp()[
                            node_index, label.item()]  # [label].item()
                # Compute explanations with the following formula
                coefs[i] = (torch.abs(probas-probas_) /
                            probas).cpu().detach().numpy()

        return coefs

    def explain_nei(self, node_index=0, hops=2, num_samples=0, info=False, multiclass=False):
        """Greedy explainer - only considers node features for explanations
        """
        # Construct k hop subgraph of node of interest (denoted v)
        neighbours, _, _, edge_mask =\
            torch_geometric.utils.k_hop_subgraph(node_idx=node_index,
                                                 num_hops=hops,
                                                 edge_index=self.data.edge_index)
        # Store the indexes of the neighbours of v (+ index of v itself)

        # Remove node v index from neighbours and store their number in D
        neighbours = neighbours[neighbours != node_index]
        self.neighbours = neighbours
        self.M = neighbours.shape[0]

        # Compute predictions
        if self.gpu:
            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
            self.model = self.model.to(device)
            with torch.no_grad():
                probas = self.model(x=self.data.x.cuda(), edge_index=self.data.edge_index.cuda()).exp()[
                    node_index]
        else:
            with torch.no_grad():
                probas = self.model(x=self.data.x, edge_index=self.data.edge_index).exp()[
                    node_index]
        pred_confidence, label = torch.topk(probas, k=1)

        if multiclass:
            # Init explanations vector
            coefs = np.zeros([self.M, self.data.num_classes])  # (m, #feats)

            # Loop on all neighbours - consider all classes
            for i, nei_idx in enumerate(self.neighbours):
                nei_idx = nei_idx.item()
                A_ = deepcopy(self.data.edge_index)

                # Find all edges incident to the isolated neighbour
                pos = (self.data.edge_index == nei_idx).nonzero()[
                    :, 1].tolist()

                # Create new adjacency matrix where this neighbour is isolated
                A_ = np.array(self.data.edge_index)
                A_ = np.delete(A_, pos, axis=1)
                A_ = torch.tensor(A_)

                # Compute new prediction with updated adj matrix (without this neighbour)
                if self.gpu:
                    with torch.no_grad():
                        probas_ = self.model(x=self.data.x.cuda(), edge_index=A_.cuda()).exp()[
                            node_index]
                else:
                    with torch.no_grad():
                        probas_ = self.model(x=self.data.x, edge_index=A_).exp()[
                            node_index]
                # Compute explanations with the following formula
                coefs[i] = (torch.abs(probas-probas_) /
                            probas).cpu().detach().numpy()

        else:
            probas = probas[label.item()]
            coefs = np.zeros(self.M)
            for i, nei_idx in enumerate(self.neighbours):
                nei_idx = nei_idx.item()
                A_ = deepcopy(self.data.edge_index)

                # Find all edges incident to the isolated neighbour
                pos = (self.data.edge_index == nei_idx).nonzero()[
                    :, 1].tolist()

                # Create new adjacency matrix where this neighbour is isolated
                A_ = np.array(self.data.edge_index)
                A_ = np.delete(A_, pos, axis=1)
                A_ = torch.tensor(A_)

                # Compute new prediction with updated adj matrix (without this neighbour)
                if self.gpu:
                    with torch.no_grad():
                        probas_ = self.model(x=self.data.x.cuda(), edge_index=A_.cuda()).exp()[
                            node_index, label.item()]
                else:
                    with torch.no_grad():
                        probas_ = self.model(x=self.data.x, edge_index=A_).exp()[
                            node_index, label.item()]
                # Compute explanations with the following formula
                coefs[i] = (torch.abs(probas-probas_) /
                            probas).cpu().detach().numpy()

        return coefs