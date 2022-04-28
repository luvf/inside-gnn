from ExplanationEvaluation.evaluation.BaseEvaluation import BaseEvaluation
from ExplanationEvaluation.evaluation.utils import evaluation_auc
from torch_geometric.nn import MessagePassing
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import average_precision_score,f1_score
import numpy as np
from tqdm import tqdm
from scipy.special import softmax
class PREvaluation(BaseEvaluation):
    """
    A class enabling the evaluation of the AUC metric on both graphs and nodes.

    :param task: str either "node" or "graph".
    :param ground_truth: ground truth labels.
    :param indices: Which indices to evaluate.

    :funcion get_score: obtain the roc auc score.
    """
    def __init__(self, task, ground_truth, indices,**kwargs):
        self.task = task
        self.ground_truth = ground_truth
        self.indices = indices


    def get_score(self, explanations):
        """
        Determines the auc score based on the given list of explanations and the list of ground truths
        :param explanations: list of explanations
        :return: auc score
        """
        th= 0.5
        correct=0
        atrr=0
        app=0
        ground_truth = []
        predictions = []

        for idx, n in tqdm(enumerate(self.indices)):
            mask = explanations[idx][1].detach().numpy()
            graph = explanations[idx][0].detach().numpy()

            # Select ground truths
            edge_list = self.ground_truth[0][n]
            edge_labels = self.ground_truth[1][n]

            for edge_idx in range(0, edge_labels.shape[0]): # Consider every edge in the ground truth
                edge_ = edge_list.T[edge_idx]
                if edge_[0] == edge_[1]:  # We dont consider self loops for our evaluation (Needed for Ba2Motif)
                    continue
                t = np.where((graph.T == edge_.T).all(axis=1)) # Determine index of edge in graph

                # Retrieve predictions and ground truth
                predictions.append(mask[t][0])
                ground_truth.append(edge_labels[edge_idx])
        average_precision = average_precision_score(ground_truth, predictions)
        """        correct = ((np.array(predictions)>th) * np.array(ground_truth)).sum()
                atrr= (np.array(predictions)).sum()
                app = np.array(ground_truth).sum()
        """
        return average_precision#, f1_score(ground_truth, predictions)#correct/atrr, correct/app