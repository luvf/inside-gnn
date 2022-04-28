import torch
from torch import nn
from torch import functional as F
from ExplanationEvaluation.explainers.utils import RuleEvaluator, get_atoms
from ExplanationEvaluation.datasets.dataset_loaders import load_dataset
from torch_geometric.nn import GlobalAttention
from scipy.optimize import linear_sum_assignment
from torch_geometric.utils import to_dense_adj
from tqdm import tqdm
import networkx


class GVAE(nn.Module):
    def __init__(self, k, feature_size, model, dataset_name, adjs, features, target_rule, embedding_size=20,
                 max_node=38):
        # max_node: number of nodes in input graph, k: number of nodes in output graph
        super(GVAE, self).__init__()
        self.k = k
        self.feature_size = feature_size
        self.embedding_size = embedding_size
        self.gcn = model
        self.max_node = 38
        if max_node > k:
            self.max_node = self.k
        for param in self.gcn.parameters():
            param.requires_grad = False
        adjs = list(map(torch.tensor, adjs))
        features = list(map(torch.tensor, features))
        self.adjs, self.features = self.filter_data(adjs, features)
        self.rule_evaluator = RuleEvaluator(self.gcn, dataset_name, target_rule,
                                            ["sum", "lin", "entropy", "cheb", "likelyhood", "hamming"])
        self.target_rule = self.rule_evaluator.target_rule
        self.layer = self.target_rule[0]
        self.fc_mu = nn.Linear(self.embedding_size * self.max_node, self.max_node)
        self.fc_var = nn.Linear(self.embedding_size * self.max_node, self.max_node)
        self.gat_mu = GlobalAttention(self.fc_mu)
        self.gat_var = GlobalAttention(self.fc_var)
        # The first "max_nodes"s components are elements of mu. The rest is var.
        self.decoder_input = nn.Linear(self.embedding_size + self.max_node,
                                       self.k * (self.max_node + self.embedding_size))

        self.decoder_f1 = nn.Sequential(
            nn.Linear(self.k * (self.max_node + self.embedding_size),
                      3 * (k // 3) * (self.max_node + self.embedding_size) // 3),
            nn.ReLU(),
        )

        self.decoder_f2 = nn.Sequential(
            nn.Linear(3 * (k // 3) * (self.max_node + self.embedding_size) // 3,
                      3 * (k // 3) * (self.max_node + self.embedding_size) // 3),
            nn.ReLU(),
        )

        self.decoder_output_adj = nn.Sequential(
            nn.Linear(in_features=3 * (k // 3) * (self.max_node + self.embedding_size) // 3,
                      out_features=self.k * self.k),
            nn.Sigmoid()
        )
        self.decoder_output_feature = nn.Sequential(
            nn.Linear(in_features=3 * (k // 3) * (self.max_node + self.embedding_size) // 3,
                      out_features=self.k * self.feature_size),
            nn.Softmax()
        )

    def encoder(self, X, A):
        """
        :param X: adj matrix of size (max_node,max_node)
        :param A: feat matrix of size (max_node, feat_size)
        :return: mu, var, embed. mu, var are vectors of length max_node
        """
        embed = self.gcn.embeddings(A, X)[self.layer]
        mu, var = self.fc_mu(torch.flatten(embed)), self.fc_var(torch.flatten(embed))
        return mu, var, embed

    def decoder(self, z, rule_vector):
        """
        :param z: vector of shape (1,max_node)
        :param rule_vector: vector of shape (1,embedding_size)
        :return:
        """
        input_vector = torch.cat((z, rule_vector))
        res = self.decoder_input(input_vector)
        # res = res.view(self.k, self.k * (self.embedding_size + self.max_node))
        res = self.decoder_f1(res)
        res = self.decoder_f2(res)
        adj_prob = self.decoder_output_adj(res)
        feature_prob = self.decoder_output_feature(res)
        adj_prob = adj_prob.view(self.k, -1)
        feature_prob = feature_prob.view(self.k, -1)
        return adj_prob, feature_prob

    def forward(self, X, A):
        """
        :param X: input adjacency matrix
        :param A: input feature matrix
        :param rule: rule in binary representation
        :return: matrix X^*
        """
        mu, var, embed = self.encoder(X, A)
        z = self.reparameterize(mu, var)
        # z = z.transpose(0, 1)
        # z = z.view(z.size(1))
        rule_vector = embed[0]
        rule_vector.apply_(lambda x: bool(x))
        return *self.decoder(z, rule_vector), mu, var

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def reshape_graph(self, adj, feature):
        max_real_node = max(map(max, filter(lambda x: x[0] != x[1], adj.transpose(0, 1))))
        if max_real_node >= self.max_node:
            return None, None
        return adj[:, (adj < self.max_node).all(axis=0)], feature[:self.max_node, ]

    def filter_data(self, adjs, features):
        mask = []
        ret_adj, ret_features = [], []
        for i in range(len(adjs)):
            adj, feature = self.reshape_graph(adjs[i], features[i])
            if adj is not None:
                adjs[i], features[i] = adj, feature
                ret_adj.append(adj)
                ret_features.append(feature)
            else:
                mask.append(False)
        return ret_adj, ret_features

    @staticmethod
    def _compute_similarity(input_adj, input_feature, output_adj, output_feature, i, j, a, b):
        """
        :param input_adj:
        :param input_feature:
        :param output_adj:
        :param output_feature:
        :param i: the first node from input graph
        :param j: the second node from input graph
        :param a: the first node from output graph
        :param b: the second node from output graph
        :return: similarity function in section 3.4
        """
        try:
            return input_adj[i][j] * output_adj[a][b] * output_adj[a][a] * output_adj[b][b] * bool(
                i != j and a != b) + torch.dot(input_feature[i], output_feature[a]) * output_adj[a][a] * bool(
                i == j and a == b)
        except:
            pass

    def compute_summation(self, input_adj, input_feature, output_adj, output_feature, i, a, X_old):
        """
        :param input_adj:
        :param input_feature:
        :param output_adj:
        :param output_feature:
        :param i: node from the input graph
        :param a: node from the output graph
        :return:
        """
        res = 0
        for j in range(self.max_node):
            # j is in neighborhood of i
            # l is in neighborhood of a
            max_value = -10000000
            for l in range(self.k):
                t = self._compute_similarity(input_adj, input_feature, output_adj, output_feature, i, j, a, l) * \
                    X_old[j][l]
                max_value = max(max_value, t)
            res += max_value
        return res

    def graph_matching(self, input_adj, input_feature, output_adj, output_feature):
        X = torch.empty((self.k, self.max_node))
        X.fill_(1 / self.k)
        for _ in range(10):
            X_old = X.detach().clone()
            for i in range(self.k):
                for j in range(self.max_node):
                    X[i][j] = X_old[i][j] * self._compute_similarity(output_adj, output_feature, input_adj,
                                                                     input_feature, i, i, j,
                                                                     j) + self.compute_summation(input_adj,
                                                                                                 input_feature,
                                                                                                 output_adj,
                                                                                                 output_feature, i, j,
                                                                                                 X_old)
        return X

    @staticmethod
    def _descretize_X(X):
        row_ind, col_ind = linear_sum_assignment(-(X.detach().numpy()))
        res = torch.zeros_like(X)
        for i in range(len(col_ind)):
            res[row_ind[i]][col_ind[i]] = 1
        return res

    def loss(self, input_adj, input_feature, output_adj, output_feature, mu, var):
        input_adj = to_dense_adj(input_adj)[0]
        # kl_loss = torch.mean(-0.5 * torch.sum(1 + var - mu ** 2 - var.exp(), dim=1), dim=0)
        kl_loss = 0.5 * (-torch.sum(var) - len(var) + torch.sum(var.exp()) - torch.dot(mu, mu))
        X = self.graph_matching(input_adj, input_feature, output_adj, output_feature)
        X = self._descretize_X(X)
        A_prime = torch.matmul(torch.matmul(X, input_adj), torch.transpose(X, 0, 1))
        F_prime = torch.matmul(torch.transpose(X, 0, 1), output_feature)
        adj_loss = 0
        for a in range(self.k):
            adj_loss += A_prime[a][a] + torch.log(output_adj[a][a]) + (1 - A_prime[a][a]) * (
                    1 - torch.log(output_adj[a][a]))
        for a in range(self.k):
            for b in range(self.k):
                if a != b:
                    adj_loss += (A_prime[a][b] + torch.log(output_adj[a][b]) + (1 - A_prime[a][b]) * (
                            1 - torch.log(output_adj[a][b]))) / (self.k - 1)
        adj_loss /= self.k
        feature_loss = 0
        for a in range(self.max_node):
            t = torch.log(torch.dot(input_feature[a], F_prime[a])) if any(input_feature[a]) else 0
            feature_loss += t / self.max_node
        return -feature_loss - adj_loss + kl_loss


def train(model):
    # input is train data not test and train data
    optim = torch.optim.Adam(model.parameters())
    train_loss = 0
    for i in tqdm(range(len(model.adjs))):
        optim.zero_grad()
        adj, feature = model.adjs[i], model.features[i]
        adj_prob, feat_prob, mu, var = model(adj, feature)
        loss = model.loss(adj, feature, adj_prob, feat_prob, mu, var)
        loss.backward()
        train_loss += loss.detach().item()
        optim.step()
        print(f"loss: {train_loss}")
    torch.save(model, "~/model")


# So far decoder and encoder is done. Parameter done, train done, reshape input done, loss done, and graph matching remains done

def decode_rule(model: GVAE, rule):
    random_vector_size = model.max_node
    random_vector = torch.normal(0, 1, size=(1, random_vector_size))
    adj_prob, feature_prob = model.decoder(random_vector, rule)
    mask = []
    for i in range(len(adj_prob)):
        if adj_prob[i][i] >= 0.5:
            mask.append(True)
        else:
            mask.append(False)

    adj_prob = adj_prob[mask]
    adj_prob = torch.transpose(torch.transpose(adj_prob, 0, 1)[mask], 0, 1)
    for i in range(len(adj_prob)):
        adj_prob[i][i] = 0
        for j in range(i + 1, len(adj_prob)):
            adj_prob[j][i] = adj_prob[i][j]
    graph = networkx.convert_matrix.from_numpy_matrix(adj_prob.detach().numpy())
    feature_prob = feature_prob[mask]
    for i in range(len(feature_prob)):
        max_prob = torch.max(feature_prob[i])
        for j in range(len(feature_prob[i])):
            if feature_prob[i][j] == max_prob:
                feature_prob[i][j] = 1
            else:
                feature_prob[i][j] = 0
    tree = networkx.from_edgelist(list(networkx.algorithms.tree.mst.maximum_spanning_edges(graph)))
    return tree, feature_prob
