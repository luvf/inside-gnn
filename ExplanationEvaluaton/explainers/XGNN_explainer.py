import networkx as nx
import torch
import torch.nn as nn

import numpy as np
import copy
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import defaultdict
from .policy_nets import PolicyNN
from torch_geometric.utils import dense_to_sparse
import networkx
from tqdm import tqdm
from torch import as_tensor
from torch.nn.functional import one_hot, softmax
from torch_geometric.utils import dense_to_sparse


from ExplanationEvaluation.explainers.utils import RuleEvaluator, get_atoms

from collections import defaultdict
class gnn_explain():
    def __init__(self, model_to_explain, dataset, max_node, max_step, target_class, dataset_name, random=False, target_rule=None, target_metric="sum", real_ratio=None, edge_probs=None):
        #print('Start training pipeline')

        self.gnnNets = model_to_explain
        self.dset = dataset

        self.graph= nx.Graph()
        #$     self.mol = Chem.RWMol()  ### keep a mol obj, to check if the graph is valid
        self.max_node = max_node
        self.max_step = max_step
        self.num_class = 2
        self.learning_rate = 0.01
        self.roll_out_alpha = 2
        self.roll_out_penalty = -0.1
        self.reward_stepwise = 0.1
        self.target_class = target_class
        self.criterion = nn.CrossEntropyLoss()
        self.dset_name = dataset_name
        self.dict =defaultdict(lambda  : "x")
        self.losses =list()
        self.rewards = list()
        self.budget_gnn= 0
        self.budget_validity = 0

        if dataset_name == "ba2":
            self.unlabeled = True
        else :
            self.unlabeled = False


        self.real_ratio= real_ratio[0]
        self.prop_ratio= real_ratio[1]
        self.atoms = get_atoms(dataset_name, dataset, self.unlabeled)

        if edge_probs is not None:
            self.edge_probs = edge_probs
            keep_indexes = np.where(edge_probs["node_probs"] > 100)[0].tolist()
            self.atoms_to_consider = [self.atoms[i] for i in keep_indexes]
            self.atom_frequency =keep_indexes
        else:
            edge = np.ones((dataset[1].shape[-1], dataset[1].shape[-1]))
            degre = np.ones((dataset[1].shape[-1], 20))
            self.edge_probs = {"edge_prob": edge, "degre_prob": degre}
            self.atoms_to_consider = list(self.atoms.values())

        #self.dict.update({0:'C', 1:'N', 2:'O', 3:'F', 4:'I', 5:'Cl', 6:'Br'})
        self.random_policy = random
        #self.rule_evaluator = RuleEvaluator(self.gnnNets, dataset_name, dataset, target_rule, target_metric, atom_labels=False)
        self.rule_evaluator = RuleEvaluator(self.gnnNets, dataset_name, dataset, target_rule, target_metric, unlabeled= self.unlabeled, edge_probs=self.edge_probs)

        self.atoms = self.rule_evaluator.atoms


        self.dict.update(self.atoms)
        self.color=defaultdict(lambda  : "y")
        self.color.update({0:'g', 1:'r', 2:'b', 3:'c', 4:'m', 5:'w', 6:'y'})
        self.revatoms = self.rule_evaluator.revatoms
        self.revatoms_consider = {el:i for i, el in enumerate(self.atoms_to_consider)}

        #self.max_poss_degree = defaultdict(lambda  : 4) #{0: 4, 1: 5, 2: 2, 3: 1, 4: 7, 5:7, 6: 5}.u
        self.max_poss_degree0= {0: 4, 1: 2, 2: 3, 3: 1, 4: 1, 5: 6, 6: 2, 7: 3, 8: 1, 9: 1, 10: 3, 11: 1, 12: 1, 13: 4,
                                14: 2, 15: 1, 16: 3, 17: 3, 18: 2, 19: 4, 20: 1, 21: 4, 22: 1, 23: 2, 24: 6, 25: 6,
                                26: 2, 27: 3, 28: 2, 29: 4, 30: 3, 31: 2, 32: 3, 33: 2, 34: 3, 35: 3, 36: 3, 37: 4, 38:1,
                                39:2}

        #self.max_poss_degree = {k:self.max_poss_degree0[self.revatoms0[atom]] for k,atom in self.atoms.items() }

        #self.node_type = len(self.atoms_to_consider)
        self.node_consider = len(self.atoms_to_consider)

        #self.gnnNets = gnns.DisNets(self.node_type)
        #self.policyNets= PolicyNN(self.node_type, self.node_type, random_policy=self.random_policy)
        self.policyNets= PolicyNN(self.node_consider, self.node_consider, random_policy=self.random_policy)


        self.optimizer = optim.SGD(self.policyNets.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=5e-4)


        self.rules = self.load_rules(dataset_name)
        self.target_rule = self.rules[target_rule]

        self.target_metric = target_metric

        #self.optimizer = optim.Adam(self.policyNets.parameters(), lr=self.learning_rate)

        #"self.roll_out = Roll_out(self, self.gnnNets)

    def train_epoch(self):

        self.graph_reset()
        self.metrics = {"rollouts": list(),
                        "reward_cls": list(),
                        "reward_rollout": list(),
                        "tot_reward": list(),
                        "sucsessadd": list(),
                        "fail_add": list(),
                        "loss": list()
                        }
        for j in range(self.max_step):
            self.optimizer.zero_grad()
            reward_pred = 0
            reward_step = 0
            n = self.graph.number_of_nodes()
            if (n > self.max_node):
                break
            self.graph_old = copy.deepcopy(self.graph)
            ###get the embeddings
            X, A = self.read_from_graph(self.graph)

            #     print('Current have', n, 'node')
            X = torch.from_numpy(X)
            A = torch.from_numpy(A)
            ### Feed to the policy nets for actions
            #   forward(self, node_feat, n2n_sp, node_num):
            # start_action,tail_action, full_logits_ori = self.policyNets(X.float(), A.float(), n+self.node_type)
            start_action, start_logits_ori, tail_action, tail_logits_ori = self.policyNets(X.float(), A.float(),  n + self.node_consider)
            self.budget_gnn+=1
            if (tail_action >= n):  ####we need add node, then add edge
                if (n == self.max_node):
                    flag = False
                else:
                    self.add_node(self.graph, n, tail_action.item() - n)
                    flag = self.add_edge(self.graph, start_action.item(), n)
            else:
                flag = self.add_edge(self.graph, start_action.item(), tail_action.item())

            if flag == True :
                validity = self.check_validity(self.graph)

            if flag == True :  #### add edge  successfully
                if validity == True:
                    reward_step = self.reward_stepwise

                    reward_pred, detailed = self.get_reward(self.graph)
                    ### Then we need to roll out.

                    reward_rollout = []
                    for roll in range(10):
                        reward_cur = self.roll_out(self.graph, j)
                        reward_rollout.append(reward_cur)
                    reward_avg = torch.mean(torch.stack(reward_rollout))
                    ###desgin loss
                    total_reward = reward_step + reward_pred + reward_avg * self.roll_out_alpha  ## need to tune the hyper-parameters here.

                    if total_reward < 0:
                        self.graph = copy.deepcopy(self.graph_old)  ### rollback
                    #   total_reward= reward_step+reward_pred
                    loss = total_reward * (self.criterion(start_logits_ori[None, :], start_action.expand(1))
                                           + self.criterion(tail_logits_ori[None, :], tail_action.expand(1)))
                else:
                    total_reward = -1  # graph is not valid
                    self.graph = copy.deepcopy(self.graph_old)
                    loss = total_reward * (self.criterion(start_logits_ori[None, :], start_action.expand(1))
                                           + self.criterion(tail_logits_ori[None, :], tail_action.expand(1)))
            else:
                # ### case adding edge not successful
                ### do not evalute
                #    print('Not adding successful')
                reward_step = -1
                total_reward = reward_step + reward_pred
                #                   #print(start_logits_ori)
                # print(tail_logits_ori)
                loss = total_reward * (self.criterion(start_logits_ori[None, :], start_action.expand(1)) +
                                       self.criterion(tail_logits_ori[None, :], tail_action.expand(1)))
                #    total_reward= reward_step+reward_pred
                #     loss = total_reward*(self.criterion(stop_logits[None,:], stop_action.expand(1)) + self.criterion(start_logits_ori[None,:], start_action.expand(1)) + self.criterion(tail_logits_ori[None,:], tail_action.expand(1)))
            if not self.random_policy:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policyNets.parameters(), 100)
                self.optimizer.step()
        # self.graph_draw(self.graph)
        # plt.show()



    def train(self,budget):
        ####given the well-trained model
        ### Load the model
        self.max_prob=0

        metrics = {"rollouts" : list(),

                        "reward_cls":list(),
                        "reward_rollout":list(),
                        "tot_reward": list(),
                        "sucsessadd":list(),
                        "fail_add": list(),
                        "loss": list()
                        }
        self.log= defaultdict(list)
        graphs = list()
        steps= 0
        while budget>self.budget_gnn:# i in (range(iters)):
            steps+=1
            self.train_epoch()
            graphs.append(copy.deepcopy(self.graph))

        return graphs# self.graph

    def graph_draw(self, graph):
        attr = nx.get_node_attributes(graph, "label")
        labels = {}
        color = ''
        for n in attr:
            labels[n]= self.dict[attr[n]]
            color = color+ self.color[attr[n]]

        #   labels=dict((n,) for n in attr)
        nx.draw(graph,labels=labels, node_color=list(color))

    def check_validity(self, graph):
        return True
        self.budget_validity +=1
        node_types = nx.get_node_attributes(graph,'label')
        for i in range(graph.number_of_nodes()):
            degree = graph.degree(i)
            max_allow = self.max_poss_degree[node_types[i]]
            if(degree> max_allow):
                return False
        return True



    def check_validity2(self, graph):
        target_layer = 2
        embedding= self.gnnNets.embedding(graph)
        emb = embedding[target_layer]

        dists =0
        for e in emb:
            dists+=((self.embeddings - e)**2).sum(axis=1).min()
        return dists

    def activation_matrix(self):
        embeddings = list()
        for g, x,_ in None:
            embeddings.append(self.gnnNets.embeddings(g, x))
        self.embeddings = np.array(embeddings)


    def roll_out(self, graph, j):
        cur_graph = copy.deepcopy(graph)
        step = 0
        while(cur_graph.number_of_nodes()<=self.max_node and step<self.max_step-j):
            #  self.optimizer.zero_grad()
            step = step + 1
            X, A = self.read_from_graph(cur_graph)
            n = cur_graph.number_of_nodes()
            X = torch.from_numpy(X)
            A = torch.from_numpy(A)
            #start_action,tail_action, ori = self.policyNets(X.float(), A.float(), n+self.node_type)
            start_action, start_logits_ori, tail_action, tail_logits_ori  = self.policyNets(X.float(), A.float(), n+self.node_consider)
            self.budget_gnn+=1
            if(tail_action>=n): ####we need add node, then add edge
                #if(tail_action.item()>=n): ####we need add node, then add edge
                if(n==self.max_node):
                    flag = False
                else:
                    self.add_node(cur_graph, n, tail_action.item()-n)
                    flag = self.add_edge(cur_graph, start_action.item(), n)
            else:
                flag= self.add_edge(cur_graph, start_action.item(), tail_action.item())

            ## if the graph is not valid in rollout, two possible solutions
            ## 1. return a negative reward as overall reward for this rollout  --- what we do here.
            ## 2. compute the loss but do not update model parameters here--- update with the step loss togehter.
            if flag == True:
                validity = self.check_validity(cur_graph)
                if validity == False:
                    self.metrics["rollouts"].append(0)
                    return torch.tensor(self.roll_out_penalty)
                    #cur_graph = copy.deepcopy(graph_old)
                    # total_reward = -1
                    # loss = total_reward*(self.criterion(start_logits_ori[None,:], start_action.expand(1))
                    #             + self.criterion(tail_logits_ori[None,:], tail_action.expand(1)))
                    # self.optimizer.zero_grad()
                    # loss.backward()
                    # torch.nn.utils.clip_grad_norm_(self.policyNets.parameters(), 100)
                    ## self.optimizer.step()
            else:  ### case 1: add edges but already exists, case2: keep add node when reach max_node
                self.metrics["rollouts"].append(0)
                step= self.max_step
                #return torch.tensor(self.roll_out_penalty)
                # reward_step = -1
                # loss = reward_step*(self.criterion(start_logits_ori[None,:], start_action.expand(1)) +
                #         self.criterion(tail_logits_ori[None,:], tail_action.expand(1)))
                # self.optimizer.zero_grad()
                # loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.policyNets.parameters(), 100)
                ## self.optimizer.step()

        ###Then we evaluate the final graph

        return torch.tensor(self.get_reward(cur_graph)[0])



    def read_from_graph(self, graph):
        ## read graph with added  candidates nodes
        n = graph.number_of_nodes()
        #   degrees = [val for (node, val) in self.graph.degree()]
        #self.atoms_to_consider
        F = np.zeros((self.max_node+self.node_consider, self.node_consider))
        attr = nx.get_node_attributes(graph, "label")
        #attr = list(attr.values())
        attr = [self.revatoms_consider[el] for el in attr.values()]
        nb_clss  = self.node_consider
        targets=np.array(attr).reshape(-1)
        one_hot_feature = np.eye(nb_clss)[targets]
        F[:n,:]= one_hot_feature
        ### then get the onehot features for the candidates nodes
        F[n:n+self.node_consider,:]= np.eye(self.node_consider)

        E = np.zeros([self.max_node+self.node_consider, self.max_node+self.node_consider])
        E[:n,:n] = np.asarray(nx.to_numpy_matrix(graph))
        E[:self.max_node+self.node_consider,:self.max_node+self.node_consider] += np.eye(self.max_node+self.node_consider)
        return F, E


    def read_from_graph_raw(self, graph): ### do not add more nodes

        n = graph.number_of_nodes()
        #  F = np.zeros((self.max_node+self.node_type, 1))
        attr = nx.get_node_attributes(graph, "label")
        #attr = list(attr.values())
        attr = [self.revatoms_consider[el] for el in attr.values()]

        nb_clss  = self.node_type
        targets=np.array(attr).reshape(-1)
        one_hot_feature = np.eye(nb_clss)[targets]
        #  F[:n+1,0] = 1   #### current graph nodes n + candidates set k=1 so n+1

        E = np.zeros([n, n])
        E[:n,:n] = np.asarray(nx.to_numpy_matrix(graph))
        #   E[:n,:n] += np.eye(n)

        return one_hot_feature, E

    def graph_reset(self):
        self.graph.clear()
        atom = self.atoms_to_consider[np.random.randint(len(self.atoms_to_consider))]

        self.graph.add_node(0, label=atom)  #self.dict = {0:'C', 1:'N', 2:'O', 3:'F', 4:'I', 5:'Cl', 6:'Br'}

        # self.graph.add_edge(1, 3)
        self.step = 0
        return

    def add_node(self, graph, idx, node_type):
        graph.add_node(idx, label=self.atoms_to_consider[node_type])
        return

    def add_edge(self, graph, start_id, tail_id):
        if graph.has_edge(start_id, tail_id) or start_id == tail_id:
            return False
        else:
            graph.add_edge(start_id, tail_id)
            return True

    def compute_feature_matrix(self, graph):
        indices = []

        for node in graph.nodes():
            #index = next(filter(lambda x: self.atoms[x] == labels[node], self.atoms.keys()))

            indices.append(self.revatoms[self.nodes[node]["label"]])

        index_tensor = torch.as_tensor(indices)
        return one_hot(index_tensor, len(self.atoms.keys()))  # len(self.atoms.keys()))

    def get_embedding(self, graph, layer):
        X = self.compute_feature_matrix(graph).type(torch.float32)
        A = torch.from_numpy(networkx.convert_matrix.to_numpy_array(graph))
        A = dense_to_sparse(A)[0]
        embeddinng = self.gnnNets.embeddings(X, A)[layer][0]
        return embeddinng


    def get_reward(self, graph):
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
            score_all = self.rule_evaluator.compute_score(graph)
            if self.dset_name == "ba2":
                metric_value = -1
                real_value = -1
                score = (score_all - self.real_ratio[0]) / (self.real_ratio[1] - self.real_ratio[0])
            else:
                real = self.rule_evaluator.real_score(graph)
                mi_real = -20
                ma_rear = -2#np.exp(-1)
                if real >-2: #clamp(should not happend  »
                    real = -2
                metric_value = score_all

                real_value = real

                score = (1-self.prop_ratio)*(metric_value-self.real_ratio[0])/(self.real_ratio[1]-self.real_ratio[0]) \
                        + self.prop_ratio*(real-mi_real)/(ma_rear-mi_real)

        return score, (metric_value, real_value)

    def load_rules(self, dataset):
        names = {"ba2": ("ba2"),
                 "aids": ("Aids"),
                 "BBBP": ("Bbbp"),
                 "mutag": ("Mutag"),
                 "DD": ("DD"),
                 "PROTEINS_full":("Proteins")
                 }
        name = names[dataset]

        file = "ExplanationEvaluation/datasets/activations/" + name + "/" + name + "_activation_encode_motifs.csv"
        #file = "/home/ata/ENS de Lyon/Internship/Projects/MCTS/inter-compres/INSIDE-GNN/data/Aids/Aids_activation_encode_motifs.csv"
        rules = list()
        with open(file, "r") as f:
            for l in f:
                r = l.split("=")[1].split(" \n")[0]
                label = int(l.split(" ")[3].split(":")[1])
                rules.append((label, r))
        out = list()
        for _, r in rules:
            c = r.split(" ")
            layer = int(c[0][2])
            components = list()
            for el in c:
                components.append(int(el[5:]))
            out.append((layer, components))

        return out

    def plot_metrics(self, metrics):
        for k, v in metrics.items():
            plt.plot(v)
            plt.xlabel("epoch")

            plt.ylabel(k)
            plt.title("inter_compress")
            plt.show()




