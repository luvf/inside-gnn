import networkx as nx
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import copy
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import defaultdict
from .policy_nets import PolicyNN
from torch_geometric.utils import dense_to_sparse

class random_explainer():
    def __init__(self, model_to_explain, dset, max_node, max_step, target_class, dset_name, random=False):
        print('Start training pipeline')
        self.gnnNets = model_to_explain
        self.dset = dset

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
        self.dset_name = dset_name
        self.dict =defaultdict(lambda  : "x")
        self.losses =list()
        self.rewards = list()
        self.budget_gnn= 0
        self.budget_validity = 0



        #self.dict.update({0:'C', 1:'N', 2:'O', 3:'F', 4:'I', 5:'Cl', 6:'Br'})
        self.random_policy = random
        self.atoms0 = {0: "C", 1: "O", 2: "N", 3: "Cl", 4: "F", 5: "S", 6: "Se", 7: "P", 8: "Na", 9: "I", 10: "Co", 11: "Br",
                       12: "Li", 13: "Si", 14: "Mg", 15: "Cu", 16: "As", 17: "B", 18: "Pt", 19: "Ru", 20: "K", 21: "Pd", 22: "Au",
                       23: "Te", 24: "W", 25: "Rh", 26: "Zn", 27: "Bi", 28: "Pb", 29: "Ge", 30: "Sb", 31: "Sn", 32: "Ga", 33: "Hg",
                       34: "Ho", 35: "Tl", 36: "Ni", 37: "Tb", 38:"H", 39:"Ca"}
        self.revatoms0= {v:k for k,v in self.atoms0.items()}
        #aids :
        self.atoms_aids = {0: "C", 1: "O", 2: "N", 3: "Cl", 4: "F", 5: "S", 6: "Se", 7: "P", 8: "Na", 9: "I", 10: "Co",
                       11: "Br",
                       12: "Li", 13: "Si", 14: "Mg", 15: "Cu", 16: "As", 17: "B", 18: "Pt", 19: "Ru", 20: "K", 21: "Pd",
                       22: "Au",
                       23: "Te", 24: "W", 25: "Rh", 26: "Zn", 27: "Bi", 28: "Pb", 29: "Ge", 30: "Sb", 31: "Sn",
                       32: "Ga", 33: "Hg",
                       34: "Ho", 35: "Tl", 36: "Ni", 37: "Tb"}
        self.atoms_mutag = {0: "C", 1: "O", 2: "Cl", 3: "H", 4: "N", 5: "F", 6: "Br", 7: "S", 8: "P", 9: "I", 10: "Na", 11: "K", 12: "Li", 13: "Ca"}
        all_atoms = {"mutag":self.atoms_mutag, "aids": self.atoms_aids}
        self.atoms= all_atoms[dset_name]
        self.dict.update(self.atoms)
        self.color=defaultdict(lambda  : "y")
        self.color.update({0:'g', 1:'r', 2:'b', 3:'c', 4:'m', 5:'w', 6:'y'})

        #self.max_poss_degree = defaultdict(lambda  : 4) #{0: 4, 1: 5, 2: 2, 3: 1, 4: 7, 5:7, 6: 5}.u
        self.max_poss_degree0= {0: 4, 1: 2, 2: 3, 3: 1, 4: 1, 5: 6, 6: 2, 7: 3, 8: 1, 9: 1, 10: 3, 11: 1, 12: 1, 13: 4,
                                14: 2, 15: 1, 16: 3, 17: 3, 18: 2, 19: 4, 20: 1, 21: 4, 22: 1, 23: 2, 24: 6, 25: 6,
                                26: 2, 27: 3, 28: 2, 29: 4, 30: 3, 31: 2, 32: 3, 33: 2, 34: 3, 35: 3, 36: 3, 37: 4, 38:1,
                                39:2}

        self.max_poss_degree = {k:self.max_poss_degree0[self.revatoms0[atom]] for k,atom in self.atoms.items() }

        self.node_type = len(self.atoms)

        #self.gnnNets = gnns.DisNets(self.node_type)
        self.policyNets= PolicyNN(self.node_type, self.node_type, random_policy=self.random_policy)
        self.optimizer = optim.SGD(self.policyNets.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=5e-4)

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
            start_action, start_logits_ori, tail_action, tail_logits_ori = self.policyNets(X.float(), A.float(),  n + self.node_type)

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

                    X_new, A_new = self.read_from_graph_raw(self.graph)
                    X_new = torch.from_numpy(X_new)
                    A_new = torch.from_numpy(A_new)

                    A_new  = dense_to_sparse(A_new)[0]
                    self.budget_gnn += 1

                    logits = self.gnnNets(X_new.float(), A_new)[0]
                    probs = torch.softmax(logits,0)

                    #### based on logits, define the reward
                    _, prediction = torch.max(logits, 0)

                    #if self.target_class == prediction:
                    # #   reward_pred = probs[prediction] - 0.5 #### positive reward
                    #else:
                    #reward_pred = 1 - probs[self.target_class]  ###negative reward
                    #reward_pred = (probs[prediction]-self.target_class).abs()

                    if self.target_class == prediction:
                        reward_pred = probs[prediction] - 0.5 #### positive reward
                    else:
                        reward_pred = probs[self.target_class] - 0.5  ###negative reward

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

    def train_epoch_perso(self):

        self.metrics = {"rollouts" : list(),
                        "reward_cls":list(),
                        "reward_rollout":list(),
                        "tot_reward": list(),
                        "sucsessadd":list(),
                        "fail_add": list(),
                        "loss": list()
                        }
        self.graph_reset()

        for j in range(self.max_step):

            X_new, A_new = self.read_from_graph_raw(self.graph)
            X_new = torch.from_numpy(X_new)
            A_new = torch.from_numpy(A_new)

            A_new = dense_to_sparse(A_new)[0]
            self.budget_gnn += 1

            logits = self.gnnNets(X_new.float(), A_new)[0]
            probs = torch.softmax(logits, 0)

            #### based on logits, define the reward
            _, prediction = torch.max(logits, 0)
            # if self.target_class == prediction:
            # #   reward_pred = probs[prediction] - 0.5 #### positive reward
            # else:
            # reward_pred = 1 - probs[self.target_class]  ###negative reward
            reward_pred = (probs[prediction] - self.target_class).abs()



                    ### Then we need to roll out.

                    reward_rollout= []
                    for roll in range(10):
                        reward_cur = self.roll_out(self.graph, j)
                        reward_rollout.append(reward_cur)
                    reward_avg = torch.mean(torch.stack(reward_rollout))
                    ###desgin loss
                    total_reward = reward_step+reward_pred+reward_avg*self.roll_out_alpha  ## need to tune the hyper-parameters here.
                    if total_reward < 0:
                        self.graph = copy.deepcopy(self.graph_old) ### rollback
                    #   total_reward= reward_step+reward_pred
                    loss = total_reward*(self.criterion(start_logits_ori[None,:], start_action.expand(1))
                                         + self.criterion(tail_logits_ori[None,:], tail_action.expand(1)))
                    #loss = total_reward*(self.criterion(full_logits_ori.view(-1)[None,:], (start_action * 20 + tail_action).expand(1)))





                    self.rewards.append((reward_step,reward_pred.item(),reward_avg.item()))
                    self.metrics["sucsessadd"].append(1)
                    self.metrics["reward_cls"].append(reward_pred.item())
                    self.metrics["reward_rollout"].append(reward_avg.item())
                    self.metrics["loss"].append(loss.detach().item())

                    self.metrics["tot_reward"].append(total_reward.item())
                else:
                    total_reward = 4  # graph is not valid
                    self.metrics["tot_reward"].append(total_reward)
                    self.metrics["sucsessadd"].append(0)

                    self.graph = copy.deepcopy(self.graph_old)
                    loss = total_reward*(self.criterion(start_logits_ori[None,:], start_action.expand(1))
                                         + self.criterion(tail_logits_ori[None,:], tail_action.expand(1)))
                    #loss = total_reward*(self.criterion(full_logits_ori.view(-1)[None,:], (start_action * 20 + tail_action).expand(1)))

                    self.metrics["loss"].append(loss.detach().item())

            else:
                # ### case adding edge not successful
                ### do not evalute
                #    print('Not adding successful')
                self.metrics["fail_add"].append(1)

                reward_step = 2.
                total_reward = reward_step#+reward_pred
                self.metrics["tot_reward"].append(total_reward)

                self.rewards.append((reward_step, 0, 0))
                #                   #print(start_logits_ori)
                #print(tail_logits_ori)
                loss = total_reward*(self.criterion(start_logits_ori[None,:], start_action.expand(1)) +
                                     self.criterion(tail_logits_ori[None,:], tail_action.expand(1)))
                #loss = total_reward * (
                #    self.criterion(full_logits_ori.view(-1)[None, :], (start_action * 20 + tail_action).expand(1)))

                self.metrics["loss"].append(loss.detach().item())

                self.losses.append(loss.detach())
                #    total_reward= reward_step+reward_pred
                #loss = total_reward*(self.criterion(stop_logits[None,:], stop_action.expand(1)) + self.criterion(start_logits_ori[None,:], start_action.expand(1)) + self.criterion(tail_logits_ori[None,:], tail_action.expand(1)))
            if not self.random_policy :
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policyNets.parameters(), 100)
                self.optimizer.step()
        #self.graph_draw(self.graph)
        #plt.show()



    def train(self,iters):
        ####given the well-trained model
        ### Load the model

        metrics = {"rollouts" : list(),

                        "reward_cls":list(),
                        "reward_rollout":list(),
                        "tot_reward": list(),
                        "sucsessadd":list(),
                        "fail_add": list(),
                        "loss": list()
                        }

        for i in tqdm(range(iters)):
            self.train_epoch()
            for n,r in self.metrics.items():
                metrics[n].append(np.mean(r))
            metrs = np.array([r for r in metrics.values()])

        #self.plot_metrics(metrics)

        print(metrics)

        X_new, A_new = self.read_from_graph_raw(self.graph)
        X_new = torch.from_numpy(X_new)
        A_new = torch.from_numpy(A_new)

        A_new = dense_to_sparse(A_new)[0]
        self.budget_gnn+=1

        logits = self.gnnNets(X_new.float(), A_new)[0]
        probs = torch.softmax(logits, 0)
        prob = probs[self.target_class].item()
        print(prob)
        print("budget GNN : ", self.budget_gnn)
        print("budget validity : ", self.budget_validity)
        return self.graph

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
        self.budget_validity +=1
        node_types = nx.get_node_attributes(graph,'label')
        for i in range(graph.number_of_nodes()):
            degree = graph.degree(i)
            max_allow = self.max_poss_degree[node_types[i]]
            if(degree> max_allow):
                return False
        return True


    def add_random(self,graph):
        start_node = random.rand(len(graph))


    def roll_out(self, graph, j):
        cur_graph = copy.deepcopy(graph)
        step = 0
        while(cur_graph.number_of_nodes()<=self.max_node and step<self.max_step-j):
            #  self.optimizer.zero_grad()
            graph_old = copy.deepcopy(cur_graph)
            step = step + 1
            X, A = self.read_from_graph(cur_graph)
            n = cur_graph.number_of_nodes()
            X = torch.from_numpy(X)
            A = torch.from_numpy(A)
            #start_action,tail_action, ori = self.policyNets(X.float(), A.float(), n+self.node_type)
            start_action, start_logits_ori, tail_action, tail_logits_ori  = self.policyNets(X.float(), A.float(), n+self.node_type)
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

                return torch.tensor(self.roll_out_penalty)
                # reward_step = -1
                # loss = reward_step*(self.criterion(start_logits_ori[None,:], start_action.expand(1)) +
                #         self.criterion(tail_logits_ori[None,:], tail_action.expand(1)))
                # self.optimizer.zero_grad()
                # loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.policyNets.parameters(), 100)
                ## self.optimizer.step()

        ###Then we evaluate the final graph
        X_new, A_new = self.read_from_graph_raw(cur_graph)
        X_new = torch.from_numpy(X_new)
        A_new = torch.from_numpy(A_new)
        A_new = dense_to_sparse(A_new)[0]
        self.budget_gnn+=1

        logits = self.gnnNets(X_new.float(), A_new)[0]
        probs = torch.softmax(logits,0)

        ### Todo
        #reward = 1 - probs[self.target_class]# probs[self.target_class] - 0.5
        _, prediction = torch.max(logits, 0)

        #reward = (probs[prediction]-self.target_class).abs()
        reward = probs[self.target_class] - 0.5

        self.metrics["rollouts"].append(1)

        return reward



    def read_from_graph(self, graph): ## read graph with added  candidates nodes
        n = graph.number_of_nodes()
        #   degrees = [val for (node, val) in self.graph.degree()]
        F = np.zeros((self.max_node+self.node_type, self.node_type))
        attr = nx.get_node_attributes(graph, "label")
        attr = list(attr.values())
        nb_clss  = self.node_type
        targets=np.array(attr).reshape(-1)
        one_hot_feature = np.eye(nb_clss)[targets]
        F[:n,:]= one_hot_feature
        ### then get the onehot features for the candidates nodes
        F[n:n+self.node_type,:]= np.eye(self.node_type)

        E = np.zeros([self.max_node+self.node_type, self.max_node+self.node_type])
        E[:n,:n] = np.asarray(nx.to_numpy_matrix(graph))
        E[:self.max_node+self.node_type,:self.max_node+self.node_type] += np.eye(self.max_node+self.node_type)
        return F, E


    def read_from_graph_raw(self, graph): ### do not add more nodes
        n = graph.number_of_nodes()
        #  F = np.zeros((self.max_node+self.node_type, 1))
        attr = nx.get_node_attributes(graph, "label")
        attr = list(attr.values())
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
        self.graph.add_node(0, label= 0)  #self.dict = {0:'C', 1:'N', 2:'O', 3:'F', 4:'I', 5:'Cl', 6:'Br'}

        # self.graph.add_edge(1, 3)
        self.step = 0
        return

    def add_node(self, graph, idx, node_type):
        graph.add_node(idx, label=node_type)
        return

    def add_edge(self, graph, start_id, tail_id):
        if graph.has_edge(start_id, tail_id) or start_id == tail_id:
            return False
        else:
            graph.add_edge(start_id, tail_id)
            return True



    def plot_metrics(self, metrics):
        for k, v in metrics.items():
            plt.plot(v)
            plt.xlabel("epoch")

            plt.ylabel(k)
            plt.title("inter_compress")
            plt.show()




