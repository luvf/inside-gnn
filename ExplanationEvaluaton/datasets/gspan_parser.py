
import torch
from torch_geometric.utils import to_dense_adj, dense_to_sparse

def parse_graph(fp):
    lines=[fp.readline()]
    while len(lines[-1])>1:
        lines.append(fp.readline())

    vertex = [el.split(" ")[1:] for el in lines[:-1] if el[0]=="v"]
    edge =  [el.split(" ")[1:] for el in lines[:-1] if el[0]=="e"]
    labels=torch.ones((len(vertex),10))*0.1
    edges = torch.zeros((len(vertex),len(vertex)))
    for e in edge:
        edges[int(e[0]), int(e[1])]=1
    return dense_to_sparse(edges), labels


def gspan_parser(filename):
    out = list()
    with open(filename, "r") as f:
        l = f.readline()
        while True:
            while l[:3] != "t #":
                l = f.readline()
                if not l:
                    return out
            out.append(parse_graph(f))
            l = f.readline()
