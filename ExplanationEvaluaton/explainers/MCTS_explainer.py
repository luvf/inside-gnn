import copy
import sys
from os import system
import networkx
import networkx as nx
import numpy as np
import torch
from tqdm import tqdm
from torch import as_tensor
from torch.nn.functional import one_hot, softmax
from torch_geometric.utils import dense_to_sparse

from functools import reduce
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

from ExplanationEvaluation.explainers.utils import RuleEvaluator, get_atoms


class Node:
    def __init__(self, graph=None, parent=None, children=None, emb =None, layer=0, step=-1):
        self.graph = graph
        self.parent = parent
        self.children = children if children else list()

        self.own_value = None
        self.own_value_detailed = None
        self.activate = False

        self.value = 0
        self.visit = 0
        self.emb= emb

        self.step = step
        self.first_time = True
        self.layer = layer
        self.terminal = False

    def set_step(self,step):
        if self.step !=-1:
            print("overwritte step")
        self.step=step

    def set_own_score(self, value, evaluator=None):
        if self.own_value is not None :
            #print("overwriting value")
            return
        self.own_value = value[0]
        self.own_value_detailed = value[1]
        if evaluator is not None and self.emb is not None:
            self.activate=evaluator.activate(self.emb)
        self.visit+=1
        #self.value = value[0]


    def compute_ucb1(self):
        if self.visit == 0:
            return np.inf
        N = self.parent.visit if self.parent else 1
        return (self.value / self.visit + 1.414 * np.sqrt(np.log(N) / self.visit))

    def best_child(self, terminal=False):
        if terminal and np.all([x.terminal for x in self.children]):
            return None, None
        vals = torch.tensor([x.compute_ucb1() if  (not terminal or not x.terminal )else -np.inf for x in self.children])


        i = vals.argmax()
        return self.children[i], i
        #return reduce(lambda x, y: x if x.compute_ucb1(self) >= y.compute_ucb1(self) else y, )

    def clean_tree(self):
        self.children=list(filter( lambda x : x.emb is not None , self.children))
        list(map(lambda  x : x.clean_tree(), self.children))


    def as_list(self):
        return [self] + [el for ch in self.children for el in ch.as_list() if ch.parent==self]


class Tree:
    def __init__(self, root):
        self.root = root
        self.unique_nodes = list()

    def back_propagate(self, node, value, path=None):
        if node == self.root:
            return
        if path:
            path, parent = path[:-1], path[-1]
        else:
            parent = node.parent
        parent.value += value

        self.back_propagate(parent, value, path=path)

    def sort_nodes(self, explainer, metric, valid=True):
        nodes = self.root.as_list()
        graph_value = [(g.graph,
                        explainer.rule_evaluator.compute_score_emb(g.emb, metric),
                        explainer.rule_evaluator.real_score(g.graph))
                        for g in nodes if g.emb is not None and len(g.graph) >1 and (not valid or explainer.rule_evaluator.activate(g.emb))]
        return list(sorted(graph_value, key=lambda el: el[1], reverse=True))


class MCTSExplainer:
    def __init__(self, model_to_explain, dataset, max_node, max_edge, target_class, dataset_name, target_rule=None,
                     target_metric="sum", uid=0,edge_probs=None, real_ratio=1):
        self.gnnNets = model_to_explain
        self.dataset = dataset
        self.depth = 0
        self.step = 0
        self.graph = nx.Graph()
        # self.graph.add_node(0, label="C")
        self.nodes = [list() for _ in range(5000)]
        if edge_probs is not None:
            self.edge_probs = edge_probs
        else:
            edge =np.ones((dataset[1].shape[-1], dataset[1].shape[-1]))
            degre = np.ones((dataset[1].shape[-1], 20))
            self.edge_probs = {"edge_prob" :edge, "degre_prob": degre}
        self.tree = Tree(root=Node(graph=self.graph.copy(), step= -1))
        self.tree.root.first_time = False
        # $     self.mol = Chem.RWMol()  ### keep a mol obj, to check if the graph is valid
        self.max_edge = max_edge
        self.max_node = max_node

        self.num_class = 2

        self.target_class = target_class
        self.dataset_name = dataset_name

        self.dict = defaultdict(lambda: "x")

        self.rewards = list()
        self.DagSearch = False
        if dataset_name == "ba2":
            self.unlabeled = True
        else :
            self.unlabeled = False

        self.budget_gnn = 0
        self.budget_validity = 0



        self.real_ratio= real_ratio[0]
        self.prop_ratio= real_ratio[1]

        self.atoms = get_atoms(dataset_name, dataset, self.unlabeled)
        if edge_probs is not None :
            keep_indexes = np.where(edge_probs["node_probs"]>100)[0].tolist()
            self.atoms_to_consider = [self.atoms[i] for i in keep_indexes]
        else:
            self.atoms_to_consider = list(self.atoms.values())

        self.revatoms = {v: k for k, v in self.atoms.items()}




        self.color = defaultdict(lambda: "y")
        self.color.update({0: 'g', 1: 'r', 2: 'b', 3: 'c', 4: 'm', 5: 'w', 6: 'y'})

        """# self.max_poss_degree = defaultdict(lambda  : 4) #{0: 4, 1: 5, 2: 2, 3: 1, 4: 7, 5:7, 6: 5}.u
        self.max_poss_degree0 = {0: 4, 1: 2, 2: 3, 3: 1, 4: 1, 5: 6, 6: 2, 7: 3, 8: 1, 9: 1, 10: 3, 11: 1, 12: 1, 13: 4,
                                 14: 2, 15: 1, 16: 3, 17: 3, 18: 2, 19: 4, 20: 1, 21: 4, 22: 1, 23: 2, 24: 6, 25: 6,
                                 26: 2, 27: 3, 28: 2, 29: 4, 30: 3, 31: 2, 32: 3, 33: 2, 34: 3, 35: 3, 36: 3, 37: 4,
                                 38: 1,
                                 39: 2}

        self.max_poss_degree = {k: self.max_poss_degree0[self.revatoms[atom]] for k, atom in self.atoms.items()}
            """
        self.node_type = len(self.atoms)

        self.rule_evaluator = RuleEvaluator(self.gnnNets, dataset_name, dataset, target_rule, target_metric, unlabeled= self.unlabeled, edge_probs=self.edge_probs)
        self.target_rule = self.rule_evaluator.target_rule

        self.best_score = [0]
        self.step_score = [0]
        self.size_best = [0]
        self.target_metric = target_metric
        self.rollout_graphs = list()
        self.uid= uid


    def find_node(self, graph, emb):
        if not self.DagSearch:
            return None
        n = len(graph)
        e = graph.number_of_edges()
        v = int(emb.sum().item()*28937)%len(self.nodes)
        #tuple([n,e]+[(el >0.1).item() for el in emb[:5]])
        for node in self.nodes[v]:
            if (emb == node.emb).all():
                if nx.is_isomorphic(graph, node.graph, node_match=lambda x, y: x == y):
                    return node
        return None

    def graph_draw(self, graph):
        attr = nx.get_node_attributes(graph, "label")
        labels = {}
        color = ''
        for n in attr:
            labels[n] = attr[n]
            color = color + self.color[attr[n]]
        color = list(color)
        color[0]="r"
        #   labels=dict((n,) for n in attr)
        nx.draw(graph, labels=labels, node_color=color)

    def graph_reset(self):
        self.graph.clear()
        return

    def check_validity(self, graph):
        return True
        self.budget_validity += 1
        node_types = nx.get_node_attributes(graph, 'label')
        nodes = list(graph.nodes())
        labels = networkx.get_node_attributes(graph, 'label')
        for i in graph.nodes():
            index = next(filter(lambda x: self.atoms[x] == labels[i], self.atoms.keys()))
            degree = graph.degree(i)
            max_allow = self.max_poss_degree[index]
            if (degree > max_allow):
                return False
        return True

    def compute_feature_matrix(self, graph):
        if self.dataset_name =="ba2":
            return torch.ones(len(graph),10)*0.1
        indices = []
        labels = networkx.get_node_attributes(graph, 'label')
        for node in graph.nodes():
            index = next(filter(lambda x: self.atoms[x] == labels[node], self.atoms.keys()))
            indices.append(index)

        index_tensor = as_tensor(indices)
        return one_hot(index_tensor, len(self.atoms.keys()))  # len(self.atoms.keys()))

    def compute_score(self, graph, emb=None):
        self.budget_gnn += 1
        metric_value = -1
        real_value = -1
        if not self.target_rule:
            X = self.compute_feature_matrix(graph).type(torch.float32)
            A = torch.from_numpy(networkx.convert_matrix.to_numpy_array(graph))
            A = dense_to_sparse(A)[0]
            score = softmax(self.gnnNets(X, A)[0], 0)[self.target_class].item()
            score_all = dict()
        else:
            if emb is not None:
                score_all = self.rule_evaluator.compute_score_emb(emb)
            else:
                score_all = self.rule_evaluator.compute_score(graph)
            if self.dataset == "ba2":
                metric_value = -1
                real_value = -1
                score = (score_all - self.real_ratio[0]) / (self.real_ratio[1] - self.real_ratio[0])
            else:
                real = self.rule_evaluator.real_score(graph)
                mi_real = -20
                ma_rear = -2 #np.exp(-1)
                if real >-2: #clamp(should not happend )
                    real = -2
                metric_value = score_all

                real_value = real

                score = (1-self.prop_ratio)*(metric_value-self.real_ratio[0])/(self.real_ratio[1]-self.real_ratio[0]) \
                        + self.prop_ratio*(real-mi_real)/(ma_rear-mi_real)



        return score, (metric_value, real_value)

    def is_terminal(self, node):
        return node.graph.number_of_edges() >= self.max_edge or len(node.graph)>=  self.max_node

    def gen_children(self, node):
        graph = node.graph
        nodes = graph.nodes()
        new_nodes_offset = len(nodes)
        index = new_nodes_offset
        children = list()
        if not new_nodes_offset:
            for atom in self.atoms_to_consider:
                new_graph = copy.deepcopy(graph)
                new_graph.add_node(index, label=atom)
                new_node = Node(graph=new_graph, parent=node,layer=1)
                children.append(new_node)
        else:
            nodes = list(nodes) + self.atoms_to_consider
            layer = node.layer
            for i in range(new_nodes_offset):
                v = nodes[i]
                d1 = nx.dijkstra_path_length(graph, 0, v)
                if d1>=layer-1:
                    for j in range(i + 1, len(nodes)):
                        u = nodes[j]
                        index_u = nodes[j]
                        if j >= new_nodes_offset:
                            index_u = index
                            d2= d1+1
                        else :
                            d2 = nx.dijkstra_path_length(graph, 0, index_u)
                        if d2== layer and d1==layer:
                            d2 = layer + 1
                        if not graph.has_edge(v, index_u) and d2 >= layer-1 and self.rule_evaluator.get_layer()+1>=d2:
                            new_graph = copy.deepcopy(graph)#graph.copy()
                            if j >= new_nodes_offset:
                                new_graph.add_node(index, label=u)

                            new_graph.add_edge(v, index_u)

                            new_node = Node(graph=new_graph, parent=node, emb= None, layer=max(layer,d2))

                            children.append(new_node)
        return children



    def rollout(self, graph_old, layer):
        """

        :param graph_old:
        :param layer:
        :return:  the score
        """
        if self.end_condition(graph_old) or not self.check_validity(graph_old):
            return (0,(0,0)), graph_old
        graph = copy.deepcopy(graph_old)

        index = graph.number_of_nodes()
        end = False
        ll=layer
        while not end:
            # graph_ol = copy.deepcopy(graph)
            # while graph.number_of_edges() < self.max_edge:

            random = np.random.default_rng()
            nodes = graph.nodes()
            new_nodes_offset = len(nodes)
            v = random.integers(low=0, high=new_nodes_offset)
            d1 = nx.dijkstra_path_length(graph, 0, v)
            if d1>=ll-1:
                nodes = list(nodes) + self.atoms_to_consider
                u = random.integers(low=0, high=len(nodes))

                while u == v:
                    u = random.integers(low=0, high=len(nodes))

                if u >= new_nodes_offset:
                    d2=d1+1
                else :
                    d2 = nx.dijkstra_path_length(graph, 0,u)

                if d2 == ll and d1 == ll:
                    d2 = ll + 1
                if d2 >= ll - 1 and self.rule_evaluator.get_layer()+1 >= d2:
                    ll = max(layer,d2)
                    if u >= new_nodes_offset:
                        u -= new_nodes_offset
                        graph.add_node(index, label=self.atoms_to_consider[u])
                        index += 1
                        graph_nodes = list(graph.nodes())

                        edge = (graph_nodes[v], index - 1)

                        graph.add_edge(*edge)  # (graph_nodes[start_node_index], index - 1)
                    else:
                        graph_nodes = list(graph.nodes())
                        edge = (graph_nodes[v], index - 1)

                        graph.add_edge(*edge)  # (graph_nodes[start_node_index], graph_nodes[end_node_index])

                if self.rule_evaluator.get_layer() <= d2 or self.end_condition(graph):
                    end = True

        """graph.remove_edge(*edge)
        if add_node:
            graph.remove_node(index - 1)"""
        if len(graph)==len(graph_old) or graph.number_of_edges()==graph_old.number_of_edges() :
            return (0,(0,0)),-1
        score = self.compute_score(graph)
        self.rollout_graphs.append((graph, score[0]))
        return score, graph



    def train(self, iters):
        ####given the well-trained model
        ### Load the model
        best_value = -10e10
        best_solution = None
        best_prob = -10e10
        best_solution_prob = -10e10

        for i in (range(iters)):
            self.step = i
            #value, prob, graph, metrs = self.train_epoch()
            r = self.train_epoch()
            if r ==-1:
                print("ened in "+ str(i)+ " iteratons")
                return -1




        return best_solution

    def plot_scores(self):

        plt.plot(self.step_score, color="g")
        plt.plot(self.best_score, color="b")

        plt.show()

    def get_path(self, current, path):

        while True:
            if current.children:
                next_node, _ = current.best_child(terminal=True)
                if next_node is None:
                    current.terminal = True
                    path.pop()
                    if len(path)==0:
                        return -1
                    current = path[-1]
                else:
                    path.append(current)
                    current = next_node


            elif current.first_time and len(current.graph)!=0:#
                emb = self.rule_evaluator.get_embedding(current.graph, self.rule_evaluator.get_layer())

                new_node = self.find_node(current.graph, emb)
                if new_node is None:
                    current.emb = emb
                    if self.DagSearch:
                        ve = int(emb.sum().item() * 28937) % len(self.nodes)
                        self.nodes[ve].append(current)
                    current.first_time = False
                    current.set_step(self.step)

                    own_score = self.compute_score(current.graph, emb=current.emb)
                    current.set_own_score(own_score, self.rule_evaluator)

                    path.append(current)

                    return #current, path
                else:
                    parent = path[-1]

                    _, index =parent.best_child(terminal=True)
                    parent.children[index] = new_node
                    path.pop()

                    current = parent

                    #self.get_path(parent, path,i+1)
            else :
                return
        return #current, path




    def train_epoch(self):
        #self.graph_reset()


        current = self.tree.root
        path =[current]


        r = self.get_path(current,path)
        if r ==-1:
            return -1
        current = path[-1]
        if not current.terminal and not current.first_time and len(current.children)==0 :
            if self.is_terminal(current):
                current.terminal = True
            else :
                children = self.gen_children(current)
                np.random.shuffle(children)
                current.children += children
                if len(current.children) > 0:
                    #current = current.children[np.random.randint(0, len(current.children))]
                    current = current.children[0]

        for node in path:
            node.visit+=1

        #current.first_time = False

        score_and_prob, graph = self.rollout(current.graph, current.layer)
        if graph==-1:
            graph=current.graph


        #current.visit += 1
        value_to_propagate, detail = score_and_prob
        if not current.first_time:
            value_to_propagate+= current.own_value
            for node in path:
                node.visit += 1
        if self.DagSearch:
            self.tree.back_propagate(current, value_to_propagate, path=path)
        else:
            self.tree.back_propagate(current, value_to_propagate, path=None)


        self.depth = max(len(path), self.depth)
        return value_to_propagate, graph

    def end_condition(self, graph):
        if nx.eccentricity(graph,0) > self.target_rule[0] + 1:
            return True
        if graph.number_of_edges() >= self.max_edge:
            return True
        if len(graph) >= self.max_node:
            return True
        return False



    def get_embedding(self, graph, layer):
        X = self.compute_feature_matrix(graph).type(torch.float32)
        A = torch.from_numpy(networkx.convert_matrix.to_numpy_array(graph))
        A = dense_to_sparse(A)[0]
        embeddinng = self.gnnNets.embeddings(X, A)[layer][0]
        return embeddinng




    def plot_2_axis(self, graph_list):
        points = list()
        for g in graph_list:
            g = g.graph
            if len(g) > 0:
                emb = self.get_embedding(g, self.target_rule[0])
                torch.clamp(emb, 0, 1)
                mask = torch.zeros_like(emb)
                mask[self.target_rule[1]] = 1
                ratio = (sum(mask) / len(mask))
                mask.apply_(lambda x: 1 if x == 1 else 0)
                x = (mask * emb > 0).sum().item()
                mask.apply_(lambda x: 0 if x == 1 else 1)
                y = (mask * emb > 0).sum().item()
                points.append((x, y))
        x, y = list(zip(*points))
        plt.scatter(x, y, marker="+")
        plt.xlabel("components inside of the pattern")
        plt.ylabel("components Outside of the pattern")

        plt.show()

    def best_rollouts(self, k):
        graphs = sorted(self.rollout_graphs, key=lambda x: x[1], reverse=True)
        graphs = [el for el in graphs if len(el[0])>3]
        return graphs[:min(k, len(graphs))]



def plot_metrics(metrics, name, normalize=True, convolve=50):
    figure(figsize=(16, 12))
    plt.title("inter_compress, metric " + str(name))

    for k, v in metrics.items():
        if k not in ["value", "prob"]:
            v = np.array(v)
            if len(v):
                v = np.convolve(v, np.ones(convolve), "valid")/convolve
                if normalize:
                    v = (v-min(v))/(max(v)-min(v))
            plt.plot(v, label=k)
            plt.xlabel("epoch")
            plt.legend()
    plt.savefig("results/mcts/"+name+".png")

    plt.show()

    figure(figsize=(1, 1))


