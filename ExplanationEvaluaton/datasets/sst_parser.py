
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import pickle as pkl
from scipy.sparse import coo_matrix

import numpy as np
import numba
@numba.jit(nopython=True)
def help1(sqrt_deg, adj):
    return np.dot(np.dot(sqrt_deg, adj), sqrt_deg)
def to_sparse(graphs):
    edge_list=list()

    for g in graphs:
        edge_list.append(list())

        for i, u in enumerate(g):
            for j, v in enumerate(u):
                if g[i,j]:
                    edge_list[-1].append((i,j))
    return edge_list
from scipy import sparse
def load_sst(path_pkl, dir_path):

    graphs = get_tree(dir_path)
    labels = get_class(dir_path)
    smap = get_dictionnary(dir_path)
    graph_labels= get_graph_class(dir_path, smap, labels)

    #labels = get_node_class(graphs, smap,graph_labels, get_sentences(dir_path))
    node_label_lists = get_embedding(dir_path, graphs)

    #edge_lists = [list(sparse.csr_matrix(g)) for g in graphs]
    node_label_lists=[[el.detach().numpy() for el in x] for x in node_label_lists]
    edge_lists = to_sparse(graphs)



    #edge_lists, graph_labels, edge_label_lists, node_label_lists = graphs,
    #edge_lists, graph_labels, edge_label_lists, node_label_lists = get_graph_data(path_graph)


    graph_labels[graph_labels == -1] = 0

    #filter
    if False:
        filter=[ i for i,node_label in enumerate(graph_labels) if abs(node_label-0.5)>=0.25]
        edge_lists = np.take(edge_lists, filter)
        graph_labels = np.take(graph_labels, filter)
        edge_label_lists = np.take(edge_label_lists, filter)
        node_label_lists = np.take(node_label_lists, filter)
    graph_labels = np.array(graph_labels) > 0.5

    edge_label_lists=[[0] for _ in graphs]

    max_node_nmb = np.max([len(node_label) for node_label in node_label_lists]) + 1  # add nodes for each graph

    edge_label_nmb = np.max([np.max(l) for l in edge_label_lists]) + 1
    node_label_nmb = np.zeros(len(node_label_lists[0][0]))
    #np.max([np.max(l) for l in node_label_lists]) + 1

    for gid in range(len(edge_lists)):
        node_nmb = len(node_label_lists[gid])
        for nid in range(node_nmb, max_node_nmb):
            edge_lists[gid].append((nid, nid))  # add self edges
            node_label_lists[gid].append(node_label_nmb)  # the label of added node is node_label_nmb
            edge_label_lists[gid].append(edge_label_nmb)

    adjs = []
    for edge_list in tqdm(edge_lists):
        row = np.array(edge_list)[:, 0]
        col = np.array(edge_list)[:, 1]
        data = np.ones(row.shape)
        adj = coo_matrix((data, (row, col))).toarray()
        if True:  # originally checked the adjacency to be normal
            degree = np.sum(adj, axis=0, dtype=float).squeeze()
            degree[degree == 0] = 1
            sqrt_deg = np.diag(1.0 / np.sqrt(degree))
            adj = help1(sqrt_deg, adj)
            #adj = np.matmul(np.matmul(sqrt_deg, adj), sqrt_deg)
        adjs.append(np.expand_dims(adj, 0))

    labels = graph_labels

    adjs = np.concatenate(adjs, 0)
    labels = np.array(labels).astype(int)
    feas = []

    """for node_label in node_label_lists:
        fea = np.zeros((len(node_label), node_label_nmb + 1))
        rows = np.arange(len(node_label))
        fea[rows, node_label] = 1
        fea = fea[:, :-1]  # remove the added node feature

        if node_label_nmb < 3:
            const_features = np.ones([fea.shape[0], 10])
            fea = np.concatenate([fea, const_features], -1)
        feas.append(fea)
    feas = np.array(feas[:adjs.shape[0]])

    feas = np.array(feas)"""
    feas = np.array(node_label_lists)

    b = np.zeros((labels.size, labels.max() + 1))
    b[np.arange(labels.size), labels] = 1
    labels = b
    with open(path_pkl,'wb') as fout:
        pkl.dump((adjs, feas,labels),fout)
    return adjs, feas, labels




def get_tree(path):
    out= list()
    with open(path+"/stanfordSentimentTreebank/STree.txt", "r") as f:
        for l in f:
            l = l.split("|")
            l = [int(el) for el in l]
            graph = torch.zeros((len(l),len(l)))
            for i,el in enumerate(l[:-1]):
                graph[i,el-1]= graph[el-1,i]=1
            out.append(graph)
    return out

def get_sentences(path):
    out =list()
    with open(path+ "/stanfordSentimentTreebank/STree.txt", "r") as f:
        for l in tqdm(f):
            l = l.split("|")
            out.append(l)
    return out

def get_embedding(path, graphs):
    out= list()
    lines =list()
    with open(path+ "/stanfordSentimentTreebank/SOStr.txt", "r") as f:
        for l in f:
            lines.append(l.split("|"))
    #embedding, idx = get_embedding_fn(lines)
    #embedding, idx = get_embeddings2(lines)

    word_vectors = api.load("glove-wiki-gigaword-100")
    #test_sentence = [''.join(e for e in x if e.isalnum() and e not in ["\n",'"',' ']).lower() for el in l for x in el if x!=""]#flatten


    for i,l in enumerate(lines):
        l = [''.join(e for e in el if e.isalnum() and e not in ["\n",'"',' ']).lower() for el in l]
        l = [word_vectors[el] if el in word_vectors else np.zeros(100, dtype=float)  for el in l ]
        l = [torch.tensor(el,dtype=torch.float32) for el in l]

        #l = [embedding(torch.tensor(idx[el],dtype=torch.long)) for el in l]
        emb = list()
        j=0
        for k in range(len(graphs[i])):
            if graphs[i][k].sum() ==1:
                emb.append(l[j])
                j+=1
            else:
                emb.append(torch.zeros(100))
        out.append(emb)
    return out


def get_dictionnary(path):
    out = dict()
    with open(path+"/stanfordSentimentTreebank/dictionary.txt", "r") as f:
        for l in f:

            prop, id = l.split("|")
            out[prop]=id
    return out

def get_class(path):
    out = list()
    with open(path+"/stanfordSentimentTreebank/sentiment_labels.txt", "r") as f:
        f.readline()
        for l in f:

            prop, id = l.split("|")
            out.append(float(id))
    return out

def build_sentences(g,s):
    sentences = s + [None for _ in range(len(s),len(g))]
    for i in range(len(g)):
        build_sentences_rec(i, g,sentences)
    return sentences

def build_sentences_rec(i, g, sentences):
    if sentences[i]:
        return sentences[i]
    sentences[i] = " ".join([get_sentences_rec(j,g,sentences) for j in range(g[:i]) if g[k]])
    return sentences[i]

def get_node_class(graphs, smap, cls, sent):
    out = list()
    for g,s in zip(graphs,sent):
        sentences = build_sentences(g, s)
        out.append(list())
        for s in sentences:
            i = smap.get(s, None)
            out[-1].append(cls[int(i)] if i else -1)
    return out

def get_graph_class(path,smap, cls):
    out = list()

    with open(path+"/stanfordSentimentTreebank/datasetSentences.txt", "r") as f:
        f.readline()
        for l in f:
            s= l.split("\t")[1]
            i = smap[s[:-1]]
            out.append(cls[int(i)] if i else -1)

    return out

def get_split(path):
    out= {"1":list(),"2":list(),"3":list()}
    with open(path+"/stanfordSentimentTreebank/STree.txt", "r") as f:
        f.readline()
        for l in f:
            l = l.split("|")
            out[l[0]].append(int(l[1]))
    return out

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

def is_cuda_available(gpu=True):

    if gpu and torch.cuda.is_available():
        print("cuda enabled")
        return torch.device('cuda')
    else:
        print("cuda not available")
        return torch.device('cpu')
import gensim.downloader as api

def get_embeddings2(l):
    #tokenized_sentences = [sentence.split() for sentence in corpus]model = word2vec.Word2Vec(tokenized_sentences, min_count=1)
    #model = word2vec.Word2Vec(tokenized_sentences, min_count=1)

    """word_vectors = api.load("glove-wiki-gigaword-100")
    test_sentence = [''.join(e for e in x if e.isalnum() and e not in ["\n",'"',' ']).lower() for el in l for x in el if x!=""]#flatten
    return [word_vectors[el] for el in test_sentence if el in word_vectors else np.zeros(100)]"""


def get_embedding_fn(l):
    device = is_cuda_available()
    CONTEXT_SIZE = 2
    EMBEDDING_DIM = 10
    test_sentence = [e for el in l for e in el]#flatten
    #s2 = {el: i for i, el in enumerate(set(l2))}


    trigrams = [([test_sentence[i], test_sentence[i + 1]], test_sentence[i + 2])
            for line in l for i in range(len(line) - 2) ]
    # print the first 3, just so you can see what they look like
    #print(trigrams[:3])

    vocab = set(test_sentence)
    word_to_ix = {word: i for i, word in enumerate(vocab)}



    class NGramLanguageModeler(nn.Module):

        def __init__(self, vocab_size, embedding_dim, context_size):
            super(NGramLanguageModeler, self).__init__()
            self.embeddings = nn.Embedding(vocab_size, embedding_dim)
            self.linear1 = nn.Linear(context_size * embedding_dim, 128)
            self.linear2 = nn.Linear(128, vocab_size)

        def forward(self, inputs):
            batchs =inputs.size()[0]
            embeds = self.embeddings(inputs).view((batchs, -1))
            out = F.relu(self.linear1(embeds))
            out = self.linear2(out)
            log_probs = F.log_softmax(out, dim=1)
            return log_probs


    losses = []
    loss_function = nn.NLLLoss()
    model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    xx= model.embeddings(torch.tensor([word_to_ix[w] for w in test_sentence], dtype=torch.long))


    tensor_x = torch.tensor([[word_to_ix[w] for w in context] for context,_ in trigrams],dtype=torch.long) # transform to torch tensor

    tensor_y = torch.tensor([word_to_ix[el] for _,el in trigrams],dtype=torch.long)

    my_dataset = TensorDataset(tensor_x,tensor_y) # create your datset
    my_dataloader = DataLoader(my_dataset, batch_size=512) # create your dataloader


    for epoch in tqdm(range(60)):
        total_loss = 0
        #random.shuffle(trigrams)
        for x, y in my_dataloader :
            #for context, target in trigrams[:1000]:

            #context_idxs = torch.tensor([[word_to_ix[w] for w in context]for context,_ in trigrams], dtype=torch.long).to(device)


            model.zero_grad()


            #log_probs = model(context_idxs)
            log_probs = model(x.to(device))
            loss = loss_function(log_probs, y.to(device))

            #loss = loss_function(log_probs, torch.tensor([word_to_ix[target]for _,target in trigrams], dtype=torch.long))

            # Step 5. Do the backward pass and update the gradient
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        losses.append(total_loss)

        """embedding = nn.Embedding(len(s2), 10)
    
        #embeding.weight=nn.Parameter(torch.tensor(embedding_matrix,dtype=torch.float32))
        embedding_vec = embedding(torch.LongTensor([s2[l]for l in l2]))
    
        lstm=nn.LSTM(10,128,bidirectional=True,batch_first=True)(embedding_vec)
        return lstm"""
    print(losses)
    model = model.to("cpu")

    return model.embeddings, word_to_ix