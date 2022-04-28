from ExplanationEvaluation.evaluation.BaseEvaluation import BaseEvaluation
from ExplanationEvaluation.evaluation.utils import evaluation_auc
from torch_geometric.nn import MessagePassing

import numpy as np
from tqdm import tqdm
from scipy.special import softmax

class SparsityEvaluation(BaseEvaluation):
    """
    A class enabling the evaluation of the AUC metric on both graphs and nodes.

    :param task: str either "node" or "graph".
    :param ground_truth: ground truth labels.
    :param indices: Which indices to evaluate.

    :funcion get_score: obtain the roc auc score.
    """
    def __init__(self, task, ground_truth, indices):
        self.task = task
        self.ground_truth = ground_truth
        self.indices = indices

    def get_score(self, explanations):
        """
        Determines the auc score based on the given list of explanations and the list of ground truths
        :param explanations: list of explanations
        :return: auc score
        """
        xpls = [x.detach().numpy() for _,x in explanations if sum(x)>0]
        #nombre moyen d'arettes par graph, nombre d'arretes dans le dataset
        return np.mean([1-np.mean(x) for x in xpls])#, np.mean([y for y in xpls for y in y])
