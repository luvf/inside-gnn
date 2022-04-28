

import pandas as pd
from tqdm import tqdm
from ExplanationEvaluation.datasets.dataset_loaders import load_dataset

import pysubgroup as ps
from torch_geometric.utils import to_dense_adj, dense_to_sparse

import numpy as np
import networkx as nx

import matplotlib
import matplotlib.pyplot as plt""


def read_subgroup(filename):
    out = (list(), list())
    with open(filename, "r") as f:
        l = f.readline()
        while True:
            while l[0]!="t":
                l = f.readline()
                if not l:
                    return out

            g =parse_subgraph(f)
            out[1].append(g)
            while l[0] !="S":
                l = f.readline()
            out[0].append(int(l.split(" ")[1]))
            #while l!="\n":

    return out


def get_metrics(dataset):
    graphs, features, labels, _, _, _ = load_dataset(dataset)
    read_subgroup("filename"
                  )