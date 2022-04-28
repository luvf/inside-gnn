import pickle as pkl
import numpy as np
import os
from numpy.random.mtrand import RandomState

from ExplanationEvaluation.datasets.utils import preprocess_features, preprocess_adj, adj_to_edge_index, load_real_dataset,reload_aids
#from ExplanationEvaluation.datasets.sst_parser import load_sst


def load_graph_dataset(_dataset, shuffle=True):
    """Load a graph dataset and optionally shuffle it.

    :param _dataset: Which dataset to load. Choose from "ba2" or "mutag"
    :param shuffle: Boolean. Wheter to suffle the loaded dataset.
    :returns: np.array
    """
    # Load the chosen dataset from the pickle file.
    if _dataset == "ba2":
        dir_path = os.path.dirname(os.path.realpath(__file__))
        path = dir_path + '/pkls/' + "BA-2motif" + '.pkl'
        with open(path, 'rb') as fin:
            adjs, features, labels = pkl.load(fin)

    elif _dataset == "mutag":
        dir_path = os.path.dirname(os.path.realpath(__file__))
        path = dir_path + '/pkls/' + "Mutagenicity" + '.pkl'
        if not os.path.exists(path): # pkl not yet created
            print("Mutag dataset pickle is not yet created, doing this now. Can take some time")
            adjs, features, labels = load_real_dataset(path, dir_path + '/Mutagenicity/Mutagenicity_')
            print("Done with creating the mutag dataset")
        else:
            with open(path, 'rb') as fin:
                adjs, features, labels = pkl.load(fin)
    elif _dataset == "aids":
        dir_path = os.path.dirname(os.path.realpath(__file__))
        path = dir_path + '/pkls/' + "AIDS" + '.pkl'
        if not os.path.exists(path): # pkl not yet created
            print("AIDS dataset pickle is not yet created, doing this now. Can take some time")
            adjs, features, labels = load_real_dataset(path, dir_path + '/AIDS/AIDS_')
            print("Done with creating the AIDS dataset")
        else:
            with open(path, 'rb') as fin:
                adjs, features, labels = pkl.load(fin)
    elif _dataset[:5] == "Tox21":
        dir_path = os.path.dirname(os.path.realpath(__file__))
        path = dir_path + '/pkls/' + _dataset + '.pkl'
        if not os.path.exists(path): # pkl not yet created
            print("Tox21_AhR_training dataset pickle is not yet created, doing this now. Can take some time")
            adjs, features, labels = load_real_dataset(path, dir_path + '/'+_dataset+'/'+_dataset+'_')
            #features = np.array(features[:adjs.shape[0]])
            print("Done with creating the AIDS dataset")
        else:
            with open(path, 'rb') as fin:
                adjs, features, labels = pkl.load(fin)
                #features = np.array(features[:adjs.shape[0]])
    elif _dataset == "REDDIT-BINARY":
        dir_path = os.path.dirname(os.path.realpath(__file__))
        path = dir_path + '/pkls/' + _dataset + '.pkl'
        if not os.path.exists(path): # pkl not yet created
            print("reddit dataset pickle is not yet created, doing this now. Can take some time")
            adjs, features, labels = load_real_dataset(path, dir_path + '/'+_dataset+'/'+_dataset+'_')
            #features = np.array(features[:adjs.shape[0]])
            print("Done with creating the AIDS dataset")
        else:
            with open(path, 'rb') as fin:
                adjs, features, labels = pkl.load(fin)
                #features = np.array(features[:adjs.shape[0]])
    elif _dataset == "PROTEINS_full" or _dataset == "DD":
        dir_path = os.path.dirname(os.path.realpath(__file__))
        path = dir_path + '/pkls/' + _dataset + '.pkl'
        if not os.path.exists(path): # pkl not yet created
            print("reddit dataset pickle is not yet created, doing this now. Can take some time")
            adjs, features, labels = load_real_dataset(path, dir_path + '/'+_dataset+'/'+_dataset+'_')
            #features = np.array(features[:adjs.shape[0]])
            print("Done with creating the AIDS dataset")
        else:
            with open(path, 'rb') as fin:
                adjs, features, labels = pkl.load(fin)
                #features = np.array(features[:adjs.shape[0]])
    elif _dataset == "DBLP_v1" or _dataset =="deezer_ego_nets" :
        dir_path = os.path.dirname(os.path.realpath(__file__))
        path = dir_path + '/pkls/' + _dataset + '.pkl'
        if not os.path.exists(path): # pkl not yet created
            print("reddit dataset pickle is not yet created, doing this now. Can take some time")
            adjs, features, labels = load_real_dataset(path, dir_path + '/'+_dataset+'/'+_dataset+'_')
            #features = np.array(features[:adjs.shape[0]])
            print("Done with creating the AIDS dataset")
        else:
            with open(path, 'rb') as fin:
                adjs, features, labels = pkl.load(fin)
                #features = np.array(features[:adjs.shape[0]])
    elif _dataset[:3] == "sst":
        dir_path = os.path.dirname(os.path.realpath(__file__))
        path = dir_path + '/pkls/' + _dataset + '.pkl'
        if not os.path.exists(path): # pkl not yet created
            print("sst dataset pickle is not yet created, doing this now. Can take some time")
            adjs, features, labels = load_sst(path, dir_path) #load_real_dataset(path, dir_path + '/_'+dataset+'/'+dataset+'_')
            print("Done with creating the sst dataset")
        else:
            with open(path, 'rb') as fin:
                adjs, features, labels = pkl.load(fin)
    elif _dataset == "BBBP":
        dir_path = os.path.dirname(os.path.realpath(__file__))
        path = dir_path + '/pkls/' + "BBBP" + '.pkl'
        if not os.path.exists(path): # pkl not yet created
            print("BBBP dataset pickle is not yet created, doing this now. Can take some time")
            adjs, features, labels = load_real_dataset(path, dir_path + '/BBBP/BBBP_')
            print("Done with creating the BBBP dataset")
        else:
            with open(path, 'rb') as fin:
                adjs, features, labels = pkl.load(fin)
    else:
        print("Unknown dataset")
        raise NotImplementedError

    n_graphs = adjs.shape[0]
    indices = np.arange(0, n_graphs)
    if shuffle:
        prng = RandomState(42) # Make sure that the permutation is always the same, even if we set the seed different
        indices = prng.permutation(indices)

    # Create shuffled data
    adjs = adjs[indices]
    features = features[indices].astype('float32')
    labels = labels[indices]

    # Create masks
    train_indices = np.arange(0, int(n_graphs*0.8))
    val_indices = np.arange(int(n_graphs*0.8), int(n_graphs*0.9))
    test_indices = np.arange(int(n_graphs*0.9), n_graphs)
    train_mask = np.full((n_graphs), False, dtype=bool)
    train_mask[train_indices] = True
    val_mask = np.full((n_graphs), False, dtype=bool)
    val_mask[val_indices] = True
    test_mask = np.full((n_graphs), False, dtype=bool)
    test_mask[test_indices] = True

    # Transform to edge index
    edge_index = adj_to_edge_index(adjs)

    return edge_index, features, labels, train_mask, val_mask, test_mask


def _load_node_dataset(_dataset):
    """Load a node dataset.

    :param _dataset: Which dataset to load. Choose from "syn1", "syn2", "syn3" or "syn4"
    :returns: np.array
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    path = dir_path + '/pkls/' + _dataset + '.pkl'
    with open(path, 'rb') as fin:
        adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, edge_label_matrix  = pkl.load(fin)
    labels = y_train
    labels[val_mask] = y_val[val_mask]
    labels[test_mask] = y_test[test_mask]

    return adj, features, labels, train_mask, val_mask, test_mask


def load_dataset(_dataset, skip_preproccessing=False, shuffle=True):
    """High level function which loads the dataset
    by calling others spesifying in nodes or graphs.

    Keyword arguments:
    :param _dataset: Which dataset to load. Choose from "syn1", "syn2", "syn3", "syn4", "ba2" or "mutag"
    :param skip_preproccessing: Whether or not to convert the adjacency matrix to an edge matrix.
    :param shuffle: Should the returned dataset be shuffled or not.
    :returns: multiple np.arrays
    """
    print(f"Loading {_dataset} dataset")
    if _dataset[:3] == "syn": # Load node_dataset
        adj, features, labels, train_mask, val_mask, test_mask = _load_node_dataset(_dataset)
        preprocessed_features = preprocess_features(features).astype('float32')
        if skip_preproccessing:
            graph = adj
        else:
            graph = preprocess_adj(adj)[0].astype('int64').T
        labels = np.argmax(labels, axis=1)
        return graph, preprocessed_features, labels, train_mask, val_mask, test_mask
    else: # Load graph dataset
        return load_graph_dataset(_dataset, shuffle)

