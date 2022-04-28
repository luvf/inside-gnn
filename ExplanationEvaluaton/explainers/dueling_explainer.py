import gym
import torch
import torch.nn as nn
import numpy as np
from collections import deque
import random
from itertools import count
import torch.nn.functional as F
from tensorboardX import SummaryWriter


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()

        self.fc1 = nn.Linear(4, 64)
        self.relu = nn.ReLU()
        self.fc_value = nn.Linear(64, 256)
        self.fc_adv = nn.Linear(64, 256)

        self.value = nn.Linear(256, 1)
        self.adv = nn.Linear(256, 2)

    def forward(self, state):
        y = self.relu(self.fc1(state))
        value = self.relu(self.fc_value(y))
        adv = self.relu(self.fc_adv(y))

        value = self.value(value)
        adv = self.adv(adv)

        advAverage = torch.mean(adv, dim=1, keepdim=True)
        Q = value + adv - advAverage

        return Q

    def select_action(self, state):
        with torch.no_grad():
            Q = self.forward(state)
            action_index = torch.argmax(Q, dim=1)
        return action_index.item()


class Memory(object):
    def __init__(self, memory_size: int) -> None:
        self.memory_size = memory_size
        self.buffer = deque(maxlen=self.memory_size)

    def add(self, experience) -> None:
        self.buffer.append(experience)

    def size(self):
        return len(self.buffer)

    def sample(self, batch_size: int, continuous: bool = True):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        if continuous:
            rand = random.randint(0, len(self.buffer) - batch_size)
            return [self.buffer[i] for i in range(rand, rand + batch_size)]
        else:
            indexes = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
            return [self.buffer[i] for i in indexes]

    def clear(self):
        self.buffer.clear()


env = gym.make('CartPole-v0')
n_state = env.observation_space.shape[0]
n_action = env.action_space.n

onlineQNetwork = QNetwork().to(device)
targetQNetwork = QNetwork().to(device)
targetQNetwork.load_state_dict(onlineQNetwork.state_dict())

optimizer = torch.optim.Adam(onlineQNetwork.parameters(), lr=1e-4)

GAMMA = 0.99
EXPLORE = 20000
INITIAL_EPSILON = 0.1
FINAL_EPSILON = 0.0001
REPLAY_MEMORY = 50000
BATCH = 16

UPDATE_STEPS = 4

memory_replay = Memory(REPLAY_MEMORY)

epsilon = INITIAL_EPSILON
learn_steps = 0
writer = SummaryWriter('logs/ddqn')
begin_learn = False

episode_reward = 0


class Molecule_environement:
    def __init__(self, model, target_class, max_node, max_step):
        self.graph = nx.Graph()
        self.model =model
        self.target_class = target_class

        self.max_nodes = max_node
        self.max_step = max_step

        self.reset()

    def step(self,action):
        start_action, tail_action = action
        mol = self.graph
        if(tail_action.item()>=n): ####we need add node, then add edge
            if(n==self.max_node):
                    flag = False
            else:
                self.add_node(cur_graph, n, tail_action.item()-n)
                flag = self.add_edge(cur_graph, start_action.item(), n)
        else:
            flag= self.add_edge(cur_graph, start_action.item(), tail_action.item())

        if flag == True:
            validity = self.check_validity(self.graph)
            if validity:
                X_new, A_new = self.read_from_graph_raw(self.graph)
                X_new = torch.from_numpy(X_new)
                A_new = torch.from_numpy(A_new)
                A_new  = dense_to_sparse(A_new)[0]
                logits = self.gnnNets(X_new.float(), A_new)[0]
                probs = torch.softmax(logits,0)
                #### based on logits, define the reward
                _, prediction = torch.max(logits, 0)
                #if self.target_class == prediction:
                # #   reward_pred = probs[prediction] - 0.5 #### positive reward
                #else:
                #reward_pred = 1 - probs[self.target_class]  ###negative reward
                reward_pred = (probs[prediction]-self.target_class).abs()
                return self.graph, reward_pred*len(self.graph), self.done(self.graph)

        return 0

    def reset(self):
        self.graph.clear()
        self.graph.add_node(0, label= 0)  #self.dict = {0:'C', 1:'N', 2:'O', 3:'F', 4:'I', 5:'Cl', 6:'Br'}

        # self.graph.add_edge(1, 3)
        self.step = 0
        return

    def done(self):
        return(self.step>self.max_step or self.graph.number_of_nodes>self.max_nodes)

    def add_node(self, idx, node_type):
        self.graph.add_node(idx, label=node_type)
        return

    def add_edge(self, start_id, tail_id):
        
        if self.graph.has_edge(start_id, tail_id) or start_id==tail_id:
            return False
        else:
            graph.add_edge(start_id, tail_id)
            return True

    def check_validity(self):
        node_types = nx.get_node_attributes(self.graph, 'label')
        for i in range(self.graph.number_of_nodes()):
            degree = self.graph.degree(i)
            max_allow = self.max_poss_degree[node_types[i]]
            if (degree > max_allow):
                return False
        return True

    def graph_draw(self, graph):
        attr = nx.get_node_attributes(graph, "label")
        labels = {}
        color = ''
        for n in attr:
            labels[n]= self.dict[attr[n]]
            color = color+ self.color[attr[n]]

        #   labels=dict((n,) for n in attr)
        nx.draw(self.graph,labels=labels, node_color=list(color))


    def read_from_graph(self):  ## read graph with added  candidates nodes
        n = self.graph.number_of_nodes()
        #   degrees = [val for (node, val) in self.graph.degree()]
        F = np.zeros((self.max_node + self.node_type, self.node_type))
        attr = nx.get_node_attributes(self.graph, "label")
        attr = list(attr.values())
        nb_clss = self.node_type
        targets = np.array(attr).reshape(-1)
        one_hot_feature = np.eye(nb_clss)[targets]
        F[:n, :] = one_hot_feature
        ### then get the onehot features for the candidates nodes
        F[n:n + self.node_type, :] = np.eye(self.node_type)

        E = np.zeros([self.max_node + self.node_type, self.max_node + self.node_type])
        E[:n, :n] = np.asarray(nx.to_numpy_matrix(graph))
        E[:self.max_node + self.node_type, :self.max_node + self.node_type] += np.eye(self.max_node + self.node_type)
        return F, E


    def read_from_graph_raw(self):  ### do not add more nodes
        n = self.graph.number_of_nodes()
        #  F = np.zeros((self.max_node+self.node_type, 1))
        attr = nx.get_node_attributes(self.graph, "label")
        attr = list(attr.values())
        nb_clss = self.node_type
        targets = np.array(attr).reshape(-1)
        one_hot_feature = np.eye(nb_clss)[targets]
        #  F[:n+1,0] = 1   #### current graph nodes n + candidates set k=1 so n+1

        E = np.zeros([n, n])
        E[:n, :n] = np.asarray(nx.to_numpy_matrix(self.graph))
        #   E[:n,:n] += np.eye(n)

        return one_hot_feature, E


# onlineQNetwork.load_state_dict(torch.load('ddqn-policy.para'))
for epoch in count():

    state = env.reset()
    episode_reward = 0
    for time_steps in range(200):
        p = random.random()
        if p < epsilon:
            action = random.randint(0, 1)
        else:
            tensor_state = torch.FloatTensor(state).unsqueeze(0).to(device)
            action = onlineQNetwork.select_action(tensor_state)
        next_state, reward, done, _ = env.step(action)
        episode_reward += reward
        memory_replay.add((state, next_state, action, reward, done))
        if memory_replay.size() > 128:
            if begin_learn is False:
                print('learn begin!')
                begin_learn = True
            learn_steps += 1
            if learn_steps % UPDATE_STEPS == 0:
                targetQNetwork.load_state_dict(onlineQNetwork.state_dict())
            batch = memory_replay.sample(BATCH, False)
            batch_state, batch_next_state, batch_action, batch_reward, batch_done = zip(*batch)

            batch_state = torch.FloatTensor(batch_state).to(device)
            batch_next_state = torch.FloatTensor(batch_next_state).to(device)
            batch_action = torch.FloatTensor(batch_action).unsqueeze(1).to(device)
            batch_reward = torch.FloatTensor(batch_reward).unsqueeze(1).to(device)
            batch_done = torch.FloatTensor(batch_done).unsqueeze(1).to(device)

            with torch.no_grad():
                onlineQ_next = onlineQNetwork(batch_next_state)
                targetQ_next = targetQNetwork(batch_next_state)
                online_max_action = torch.argmax(onlineQ_next, dim=1, keepdim=True)
                y = batch_reward + (1 - batch_done) * GAMMA * targetQ_next.gather(1, online_max_action.long())

            loss = F.mse_loss(onlineQNetwork(batch_state).gather(1, batch_action.long()), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            writer.add_scalar('loss', loss.item(), global_step=learn_steps)

            if epsilon > FINAL_EPSILON:
                epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        if done:
            break
        state = next_state

    writer.add_scalar('episode reward', episode_reward, global_step=epoch)
    if epoch % 10 == 0:
        torch.save(onlineQNetwork.state_dict(), 'ddqn-policy.para')
        print('Ep {}\tMoving average score: {:.2f}\t'.format(epoch, episode_reward))




