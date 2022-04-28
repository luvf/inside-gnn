import torch
import torch.nn as nn
import scipy.sparse as sp
import numpy as np
import torch.nn.functional as F

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    #print("===Normalizing adjacency matrix symmetrically===")
    adj = adj.numpy()
    N = adj.shape[0]
    #  adj = adj + np.eye(N)
    D = np.sum(adj, 0)
    D_hat = np.diag(np.power(D,-0.5))
    #   np.diag((D )**(-0.5))
    out = np.dot(D_hat, adj).dot(D_hat)
    out[np.isnan(out)]=0
    out = torch.from_numpy(out)
    return out, out.float()

import torch
import torch.nn as nn
class GCN(nn.Module):

    def __init__(self, in_dim, out_dim, p=0.3):
        super(GCN, self).__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.drop = nn.Dropout(p=p)

        weights_init(self)

    def forward(self, A, X):
        #    X = self.drop(X)
        X = torch.matmul(A, X)
        X = self.proj(X)
        return torch.relu(X)#torch.sigmoid(X)#X#torch.relu(X)#

import numpy as np
from torch.nn.parameter import Parameter
import torch.nn as nn


def glorot_uniform(t):
    if len(t.size()) == 2:
        fan_in, fan_out = t.size()
    elif len(t.size()) == 3:
        # out_ch, in_ch, kernel for Conv 1
        fan_in = t.size()[1] * t.size()[2]
        fan_out = t.size()[0] * t.size()[2]
    else:
        fan_in = np.prod(t.size())
        fan_out = np.prod(t.size())

    limit = np.sqrt(6.0 / (fan_in + fan_out))
    t.uniform_(-limit, limit)


def _param_init(m):
    if isinstance(m, Parameter):
        glorot_uniform(m.data)
    elif isinstance(m, nn.Linear):
        m.bias.data.zero_()
        glorot_uniform(m.weight.data)
def weights_init(m):
    for p in m.modules():
        if isinstance(p, nn.ParameterList):
            for pp in p:
                _param_init(pp)
        else:
            _param_init(p)

    for name, p in m.named_parameters():
        if '.' not in name:
            _param_init(p)

class PolicyNN(nn.Module):
    def __init__(self,  input_dim, node_type_num, initial_dim=8, latent_dim=[16, 24, 32],  max_node = 12, random_policy=False):
        #print('Initializing Policy Nets')
        super(PolicyNN, self).__init__()

        self.latent_dim = latent_dim
        self.input_dim  = input_dim
        self.node_type_num =node_type_num
        self.initial_dim = initial_dim
        #      self.stop_mlp_hidden = 16
        self.start_mlp_hidden = 16
        self.tail_mlp_hidden = 24

        self.input_mlp = nn.Linear(self.input_dim, initial_dim)


        self.gcns = nn.ModuleList()
        self.layer_num = len(latent_dim)
        self.gcns.append(GCN(self.initial_dim, self.latent_dim[0]))
        for i in range(1, len(latent_dim)):
            self.gcns.append(GCN(self.latent_dim[i-1], self.latent_dim[i]))

        self.dense_dim = latent_dim[-1]

        # self.stop_mlp1 = nn.Linear(self.dense_dim, self.stop_mlp_hidden)
        # self.stop_mlp_non_linear= nn.ReLU6()
        # self.stop_mlp2 = nn.Linear(self.stop_mlp_hidden, 2)

        self.start_mlp1= nn.Linear(self.dense_dim, self.start_mlp_hidden)
        self.start_mlp_non_linear = nn.ReLU6()
        self.start_mlp2= nn.Linear(self.start_mlp_hidden, 1)


        self.tail_mlp1= nn.Linear(2*self.dense_dim, self.tail_mlp_hidden)
        self.tail_mlp_non_linear = nn.ReLU6()
        self.tail_mlp2= nn.Linear(self.tail_mlp_hidden, 1)

        self.full_mlp1 = nn.Linear(2 * self.dense_dim, self.tail_mlp_hidden)
        self.full_mlp_non_linear = nn.ReLU6()
        self.full_mlp2 = nn.Linear(self.tail_mlp_hidden, 1)

        weights_init(self)
        self.random_policy=random_policy



    def forward(self, node_feat, n2n_sp, node_num):

        un_A, A = normalize_adj(n2n_sp)
        #  A = n2n_sp
        #   print('adj has shape', A.size())
        #   print('node feature has shape', node_feat.size())
        cur_out = node_feat
        cur_A = A

        cur_out = self.input_mlp(cur_out)
        #      cur_out = self.input_non_linear(cur_out)

        for i in range(self.layer_num):
            cur_out = self.gcns[i](cur_A, cur_out)



        ### now we have the node embeddings

        ### get two different masks
        ob_len = node_num  ##total current + candidates set
        ob_len_first = ob_len - self.node_type_num

        logits_mask = self.sequence_mask(ob_len, cur_A.size()[0])

        logits_mask_first = self.sequence_mask(ob_len_first, cur_A.size()[0])
        #     print('logits_mask_first has shape', logits_mask_first.size())
        graph_embedding = torch.mean(cur_out, 0)  ####
        #  print('graph_embedding has shape', graph_embedding.size())

        ### action--- select the starting node, two layer mlps

        start_emb = self.start_mlp1(cur_out)
        start_emb = self.start_mlp_non_linear(start_emb)
        start_logits = self.start_mlp2(start_emb)
        if self.random_policy:
            start_logits = torch.rand(start_logits.shape)
        start_logits_ori = torch.squeeze(start_logits)
        #    print('start_logits has shape', start_logits.size())
        start_logits_short = start_logits_ori[0:ob_len_first]

        start_probs = torch.nn.functional.softmax(start_logits_short, dim=0)

        start_prob_dist = torch.distributions.Categorical(start_probs)
        try:
            start_action = start_prob_dist.sample()
        except:
            import pdb
            pdb.set_trace()

        mask = F.one_hot(start_action, num_classes=node_feat.size()[0])
        mask = mask.bool()
        emb_selected_node = torch.masked_select(cur_out, mask[:, None])

        ### action--- select the tail node, two layer mlps
        emb_selected_node_copy = emb_selected_node.repeat(cur_out.size()[0], 1)
        cat_emb = torch.cat((cur_out, emb_selected_node_copy), 1)

        tail_emb = self.tail_mlp1(cat_emb)
        tail_emb = self.tail_mlp_non_linear(tail_emb)
        tail_logits = self.tail_mlp2(tail_emb)
        if self.random_policy:
            tail_logits = torch.rand(start_logits.shape)
        tail_logits_ori = torch.squeeze(tail_logits)

        logits_second_mask = logits_mask[0] & ~mask
        tail_logits_short = tail_logits_ori[0:ob_len]
        logits_second_mask_short = logits_second_mask[0:ob_len]

        tail_logits_null = torch.ones_like(tail_logits_short) * -1000000
        tail_logits_short = torch.where(logits_second_mask_short == True, tail_logits_short, tail_logits_null)

        tail_probs = torch.nn.functional.softmax(tail_logits_short, dim=0)

        tail_prob_dist = torch.distributions.Categorical(tail_probs)
        try:
            tail_action = tail_prob_dist.sample()
        #            if tail_action >= start_action:
        #                tail_action = tail_action +1
        except:
            import pdb
            pdb.set_trace()

        return start_action, start_logits_ori, tail_action, tail_logits_ori


    def sequence_mask(self, lengths, maxlen, dtype=torch.bool):
        mask = ~(torch.ones((lengths, maxlen)).cumsum(dim=1).t() > lengths).t()
        mask.type(dtype)
        return mask