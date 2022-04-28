
from ExplanationEvaluation.configs.selector import Selector
from ExplanationEvaluation.tasks.training import train_node, train_graph,load_best_model,create_data_list,evaluate

from ExplanationEvaluation.datasets.dataset_loaders import load_dataset
from torch_geometric.data import Data, DataLoader
from ExplanationEvaluation.models.model_selector import model_selector

import torch
import numpy as np


def get_dset_training():
    outstr = ""
    for dataset in ["dd","proteins", "ba2motifs",  "aids","mutag" ,"BBBP"][:2] :
        _model = 'gnn'
        config_path = f"./ExplanationEvaluation/configs/replication/models/model_{_model}_{dataset}.json"

        config = Selector(config_path).args

        graphs, features, labels, train_mask, val_mask, test_mask = load_dataset(config.model.dataset)
        model = model_selector(config.model.paper, config.model.dataset, False)

        train_set = create_data_list(graphs, features, labels, train_mask)
        val_set = create_data_list(graphs, features, labels, val_mask)
        test_set = create_data_list(graphs, features, labels, test_mask)

        model = load_best_model(-1, config.model.paper, config.model.dataset, model, True)
        train_loader = DataLoader(train_set, batch_size=config.model.batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=len(val_set), shuffle=False)
        test_loader = DataLoader(test_set, batch_size=len(test_set), shuffle=False)
        with torch.no_grad():
            train_sum = 0
            for data in train_loader:
                out = model(data.x, data.edge_index, data.batch)
                preds = out.argmax(dim=1)
                train_sum += (preds == data.y).sum()
            train_acc = int(train_sum) / int(len(train_set))

            eval_data = next(iter(test_loader))
            out = model(eval_data.x, eval_data.edge_index, eval_data.batch)
            test_acc = evaluate(out, eval_data.y)

            eval_data = next(iter(val_loader))
            out = model(eval_data.x, eval_data.edge_index, eval_data.batch)
            val_acc = evaluate(out, eval_data.y)
        out = [len(graphs), labels.sum(axis=0), np.mean([el.sum() for el in features]), np.mean([(el[0]!=el[1]).sum() for el in graphs]),train_acc, test_acc, val_acc]
        out = [str(el)for el in out]
        outstr+=" & ".join(out)+"\\\\\n"
    print(outstr)


get_dset_training()

_dataset = "sst"#"tox21"#'deezer'#"proteins"'reddit' # One of: bashapes, bacommunity, treecycles, treegrids, ba2motifs, mutag

# Parameters below should only be changed if you want to run any of the experiments in the supplementary
_folder = 'replication' # One of: replication, batchnorm
_model = 'gnn' if _folder == 'replication' else 'ori'

# PGExplainer
config_path = f"./ExplanationEvaluation/configs/{_folder}/models/model_{_model}_{_dataset}.json"

config = Selector(config_path)
extension = (_folder == 'extension')


config = Selector(config_path).args

torch.manual_seed(config.model.seed)
torch.cuda.manual_seed(config.model.seed)
np.random.seed(config.model.seed)


_dataset = config.model.dataset
_explainer = config.model.paper

if _dataset[:3] == "syn":
    train_node(_dataset, _explainer, config.model)
else :# _dataset in ["deezer_ego_nets","DBLP_v1","DD","PROTEINS_full", "REDDIT-BINARY","ba2", "mutag", "aids", "Tox21_AhR_training" ,"BBBP"] :
    train_graph(_dataset, _explainer, config.model)

