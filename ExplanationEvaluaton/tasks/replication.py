import time
import json
import os

import torch
import numpy as np
from tqdm import tqdm

from ExplanationEvaluation.datasets.dataset_loaders import load_dataset
from ExplanationEvaluation.datasets.ground_truth_loaders import load_dataset_ground_truth
from ExplanationEvaluation.evaluation.AUCEvaluation import AUCEvaluation
from ExplanationEvaluation.evaluation.FidelityEvaluation import FidelityEvaluation
from ExplanationEvaluation.evaluation.InFidelityEvaluation import InFidelityEvaluation

from ExplanationEvaluation.evaluation.PREvaluation import PREvaluation
from ExplanationEvaluation.evaluation.SparsityEvaluation import SparsityEvaluation


from ExplanationEvaluation.evaluation.EfficiencyEvaluation import EfficiencyEvluation
from ExplanationEvaluation.explainers.GNNExplainer import GNNExplainer
from ExplanationEvaluation.explainers.PGExplainer import PGExplainer
from ExplanationEvaluation.explainers.InsideGNN import InsideGNN
from ExplanationEvaluation.explainers.GradExplainer import GradExplainer
from ExplanationEvaluation.explainers.PGMExplainer import PGMExplainer
from ExplanationEvaluation.explainers.SVXExplainer import GraphSVX




from ExplanationEvaluation.models.model_selector import model_selector
from ExplanationEvaluation.utils.plotting import plot


def get_classification_task(graphs):
    """
    Given the original data, determines if the task as hand is a node or graph classification task
    :return: str either 'graph' or 'node'
    """
    if isinstance(graphs, list): # We're working with a model for graph classification
        return "graph"
    else:
        return "node"


def to_torch_graph(graphs, task):
    """
    Transforms the numpy graphs to torch tensors depending on the task of the model that we want to explain
    :param graphs: list of single numpy graph
    :param task: either 'node' or 'graph'
    :return: torch tensor
    """
    if task == 'graph':
        return [torch.tensor(g) for g in graphs]
    else:
        return torch.tensor(graphs)


def select_explainer(explainer, model, graphs, features, task, epochs, lr, reg_coefs, temp=None, sample_bias=None, config=None, labels=None,**kwargs):
    """
    Select the explainer we which to use.
    :param explainer: str, "PG" or "GNN"
    :param model: graph classification model who's predictions we wish to explain.
    :param graphs: the collections of edge_indices representing the graphs
    :param features: the collcection of features for each node in the graphs.
    :param task: str "node" or "graph"
    :param epochs: amount of epochs to train our explainer
    :param lr: learning rate used in the training of the explainer
    :param reg_coefs: reguaization coefficients used in the loss. The first item in the tuple restricts the size of the explainations, the second rescticts the entropy matrix mask.
    :param temp: the temperture parameters dictacting how we sample our random graphs.
    :params sample_bias: the bias we add when sampling random graphs. 
    """
    if explainer == "PG":
        return PGExplainer(model, graphs, features, task, epochs=epochs, lr=lr, reg_coefs=reg_coefs, temp=temp, sample_bias=sample_bias)
    elif explainer == "GNN":
        return GNNExplainer(model, graphs, features, task, epochs=epochs, lr=lr, reg_coefs=reg_coefs)
    elif explainer == "ACT":
        return InsideGNN(model, graphs, features, task, config, labels, **kwargs)
    elif explainer == "CAM":
        return CamExplainer(model, graphs, features, task,**kwargs)
    elif explainer == "GRAD":
        return GradExplainer(model, graphs, features, task, **kwargs)
    elif explainer == "PGM":
        return PGMExplainer(model, graphs, features, task, **kwargs)
    elif explainer == "SVX":
        return GraphSVX(model, graphs, features, task, **kwargs)
    else:
        raise NotImplementedError("Unknown explainer type")


def run_experiment(inference_eval, evals, explainer, indices):
    """
    Runs an experiment.
    We generate explanations for given indices and calculate the AUC score.
    :param inference_eval: object for measure the inference speed
    :param auc_eval: a metric object, which calculate the AUC score
    :param explainer: the explainer we wish to obtain predictions from
    :param indices: indices over which to evaluate the auc
    :returns: AUC score, inference speed
    """
    inference_eval.start_prepate()
    explainer.prepare(indices)

    inference_eval.start_explaining()
    explanations = []
    for idx in tqdm(indices):
        graph, expl = explainer.explain(idx)
        explanations.append((graph, expl))
    inference_eval.done_explaining()

    scores = {name:el.get_score(explanations) for name, el in evals.items()}#+[inference_eval.get_score(explanations)]

    return scores


def run_qualitative_experiment(explainer, indices, labels, config, explanation_labels):
    """
    Plot the explaination generated by the explainer
    :param explainer: the explainer object
    :param indices: indices on which we validate
    :param labels: predictions of the explainer
    :param config: dict holding which subgraphs to plot
    :param explanation_labels: the ground truth labels 
    """
    for idx in indices:
        graph, expl = explainer.explain(idx)
        plot(graph, expl, labels, idx, config.thres_min, config.thres_snip, config.dataset, config, explanation_labels)


def store_results(auc, auc_std, inf_time, checkpoint, config):
    """
    Save the replication results into a json file
    :param auc: the obtained AUC score
    :param auc_std: the obtained AUC standard deviation
    :param inf_time: time it takes to make a single prediction
    :param checkpoint: the checkpoint of the explained model
    :param config: dict config
    """
    results = {"AUC": auc,
               "AUC std": auc_std,
               "Inference time (ms)": inf_time}

    model_res = {"Training Accuracy": checkpoint["train_acc"],
                 "Validation Accuracy": checkpoint["val_acc"],
                 "Test Accuracy": checkpoint["test_acc"], }

    explainer_params = {"Explainer": config.explainer,
                        "Model": config.model,
                        "Dataset": config.dataset}

    json_dict = {"Explainer parameters": explainer_params,
                 "Results": results,
                 "Trained model stats": model_res}

    save_dir = "./results"
    os.makedirs(save_dir, exist_ok=True)
    with open(f"./results/P_{config.explainer}_M_{config.model}_D_{config.dataset}_results.json", "w") as fp:
        json.dump(json_dict, fp, indent=4)


def replication(config, extension=False, run_qual=True, results_store=True,metrics=dict() , **kwargs):
    """
    Perform the replication study.
    First load a pre-trained model.
    Then we train our expainer.
    Followed by obtaining the generated explanations.
    And saving the obtained AUC score in a json file.
    :param config: a dict containing the config file values
    :param extension: bool, wheter to use all indices 
    """
    # Load complete dataset
    graphs, features, labels, _, _, test_mask = load_dataset(config.dataset)
    task = get_classification_task(graphs)

    features = torch.tensor(features)
    labels = torch.tensor(labels)
    graphs = to_torch_graph(graphs, task)

    # Load pretrained models
    model, checkpoint = model_selector(config.model,
                                        config.dataset,
                                        pretrained=True,
                                        return_checkpoint=True)
    if config.eval_enabled:
        model.eval()

    # Get ground_truth for every node
    #explanation_labels, indices = load_dataset_ground_truth(config.dataset)
    #explanation_labels =
    #indices = list(range(len(explanation_labels[0])))
    #if extension:
    #    indices = np.argwhere(test_mask).squeeze()
    explanation_labels = None
    indices=list(range(len(graphs)))

    # Get explainer
    explainer = select_explainer(config.explainer,
                                 model=model,
                                 graphs=graphs,
                                 features=features,
                                 task=task,
                                 epochs=config.epochs,
                                 lr=config.lr,
                                 reg_coefs=[config.reg_size,
                                            config.reg_ent],
                                 temp=config.temps,
                                 sample_bias=config.sample_bias,
                                 config=config,
                                 labels=labels,**kwargs)

    # Get evaluation methods
    metrs  = dict()
    for name, (metric, args) in metrics.items():
        if not (metric in [AUCEvaluation,PREvaluation]) or kwargs.get("groundtruth",False):
            if metric == FidelityEvaluation or metric == InFidelityEvaluation:
                metrs[name] = metric(task, explanation_labels, indices,model, graphs, features,dataset=config.dataset, **args)
            else:
                metrs[name] = metric(task, explanation_labels, indices, **args)

    """auc_evaluation = AUCEvaluation(task, explanation_labels, indices)
    au_evaluation = PREvaluation(task, explanation_labels, indices,**kwargs)

    fid_evaluation = FidelityEvaluation(task, explanation_labels, indices model, graphs, features,**kwargs)

    sparsity_evaluation = SparsityEvaluation(task, explanation_labels, indices, model, graphs, features)
    """
    metrs["time"] = EfficiencyEvluation()

    # Perform the evaluation 10 times
    scores = []
    times = []
    config.seeds=[4]
    for idx, s in enumerate(config.seeds):
        print(f"Run {idx} with seed {s}")
        # Set all seeds needed for reproducibility
        torch.manual_seed(s)
        torch.cuda.manual_seed(s)
        np.random.seed(s)

        metrs["time"].reset()

        score = run_experiment(metrs["time"], metrs, explainer, indices)

        if idx == 0 and run_qual: # We only run the qualitative experiment once
            run_qualitative_experiment(explainer, indices, labels, config, explanation_labels)

        scores.append(score)
        #print("score:",auc_score[0], " fid_score:", auc_score[1], " PR_score:", auc_score[2])
        #print(["metric " +str(i)+" :"+str(s) for i, s in enumerate(score)])
        #times.append(time_score)
        #print("time_elased:",time_score)

    #auc = np.mean(scores)
    #auc_std = np.std(scores)
    #inf_time = np.mean(times) / 10

    #TODO
    #if results_store: store_results(auc, auc_std, inf_time, checkpoint, config)
        
    return scores