from ExplanationEvaluation.evaluation.BaseEvaluation import BaseEvaluation
from ExplanationEvaluation.evaluation.utils import evaluation_auc
from torch_geometric.nn import MessagePassing
import torch
import numpy as np
from tqdm import tqdm
from scipy.special import softmax
from ExplanationEvaluation.explainers.InsideGNN import get_rules, explain_graph_with_motifs
import pandas as pd


class FidelityEvaluation(BaseEvaluation):
    """
    A class enabling the evaluation of the AUC metric on both graphs and nodes.

    :param task: str either "node" or "graph".
    :param ground_truth: ground truth labels.
    :param indices: Which indices to evaluate.

    :funcion get_score: obtain the roc auc score.
    """
    def __init__(self, task, ground_truth, indices, model, graphs, features, dataset="", **kwargs):
        self.task = task
        self.dataset = dataset
        self.ground_truth = ground_truth
        self.indices = indices
        self.model = model
        self.graphs = graphs
        self.features = features
        self.topK = kwargs["topK"]
        self.K = kwargs.get("K", 0)

        '''def get_xpl(self):
        file = "results/trace/fidelity/"+self.dataset+"_GNN_ACT_"+".csv"
        pertrurb_df =pd.read_csv(file)


        out= list()
        filename = 'results/trace/'+self.dataset+"_GNN_ACT_activation.csv"

        base_df= pd.read_csv(filename)
        rules = get_rules(self.dataset)
        for i,g in enumerate(self.graphs):
            out.append(explain_graph_with_motifs(pertrurb_df, base_df, g, i, rules,)[1].sum())
        return out'''

    def get_score(self, explanations):
        """
        Determines the auc score based on the given list of explanations and the list of ground truths
        :param explanations: list of explanations
        :return: auc score
        """
        #if self.topK =="variable":
        #    self.xpl = self.get_xpl()

        scores_prob= list()
        scores_bin = list()
        scores_cls= list()
        base_scores = list()
        pertrubed_scores = list()
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = False
                module.__edge_mask__ = None
        for i, indice in tqdm(enumerate(self.indices)):
            f = self.features[indice]
            g = self.graphs[indice]
            base_score1 = self.model(f,g).detach().numpy()
            base_score = softmax(base_score1)[0,1]
            base_scores.append(base_score)

            mask = explanations[i][1].detach().numpy()
            if self.topK:
                if self.topK=="variable":
                    self.K = int(self.xpl[indice].item())
                top_indices = np.argsort(mask)[-self.K:]
                top_indices = top_indices[mask[top_indices]>0]
                mask=np.zeros(mask.shape)
                mask[top_indices]=1

            mask = torch.tensor(mask,dtype=explanations[i][1].dtype)
            perturbed_score1 = self.model(f, explanations[i][0], edge_weights=1.- mask).detach().numpy()

            if np.isnan(perturbed_score1[0][0]):
                perturbed_score1 = base_score1

            perturbed_score = softmax(perturbed_score1)[0,1]
            pertrubed_scores.append(perturbed_score)


            scores_cls.append(np.abs(perturbed_score1-base_score1))
            scores_prob.append(np.abs(base_score-perturbed_score))
            scores_bin.append((base_score>0.5) != (perturbed_score >0.5))

        """        ids =np.array(su)>0
        return np.mean(np.array(scores_prob)[ids]),np.mean(np.array(scores_bin)[ids])"""

        pos = np.array(base_scores) > 0.5
        negs = np.array(base_scores) <= 0.5
        change= ((np.array(base_scores) > 0.5) != (np.array(pertrubed_scores) > 0.5))
        return np.mean(np.array(scores_prob)),np.mean(np.array(scores_bin)), change[negs].mean(), change[pos].mean()