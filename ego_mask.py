
import torch
import pandas as pd
from tqdm import tqdm

from ExplanationEvaluation.models.model_selector import model_selector
import pickle

from ExplanationEvaluation.explainers.utils import RuleEvaluator, get_atoms
import networkx


from ExplanationEvaluation.explainers.utils import get_edge_distribution
from subgraph_metrics import *

from ExplanationEvaluation.explainers.XGNN_explainer import gnn_explain
from ExplanationEvaluation.models.model_selector import model_selector
from ExplanationEvaluation.explainers.MCTS_explainer import MCTSExplainer
from ExplanationEvaluation.gspan_mine.gspan_mining.gspan import GSpanMiner
from ExplanationEvaluation.explainers.VAE import GVAE
from ExplanationEvaluation.explainers import VAE
from ExplanationEvaluation.gspan_mine.gspan_mining.gspan import GSpanMiner
from ExplanationEvaluation.explainers.RandomExplainers import RandomExplainer_subgraph, get_activation_score
from networkx.algorithms.isomorphism import ISMAGS



Number_of_rules = dict([("aids", 60), ('mutag', 60), ("BBBP", 60), ("PROTEINS_full", 28), ("DD", 47), ("ba2", 19)])

"""
def sort_nodes(nodes, value_function):
    node_list = nodes.as_list()
    return sorted([(node, value_function(node)) for node in node_list if node.visit and node.emb is not None], key= lambda x:x[1])
"""
dir_name = "mcts_dumps_new"
#dir_name = "mcts_dumps_new_small"


def run_masking(dataset, method=None, small=False):
    graphs, features, labels, _, _, _ = load_dataset(dataset)


    if dataset =="ba2":
        edge_probs=None
    else :
        if dataset =="PROTEINS_full":
            edge_probs = get_edge_distribution(graphs, features,labels,30)
        else :
            edge_probs = get_edge_distribution(graphs, features,labels)
        #degre_distrib = get_degre_distribution(graphs, features)
    model, checkpoint = model_selector("GNN",
                                       dataset,
                                       pretrained=True,
                                       return_checkpoint=True)
    metrics = [method]#, "entropy", "likelyhood_max"]
    #metrics = ["likelyhood_max"]
    atoms = get_atoms(dataset,features)
    real_ratio = {"cosine": (0, 1),
                  "entropy" : (-50,-2.014),
                  "lin" : (-1,0.1),
                  "likelyhood_max": (-20,20.04)
                  }

    rules = range( Number_of_rules[dataset])


    scores = list()
    nsteps = 5000
    nxp = 6
    print( dataset+ " "+ str(nsteps) + " " + str(nxp)+ " " + str(metrics))
    r=1
    infidelity = list()

    gdset = graph_dset([graphs,features],atoms)
    f = "results/dataframes/" + dataset + ".csv"

    x12 = MCTSExplainer(model, (graphs, features, labels), 6, 10, 1, dataset, target_rule=0,
                              target_metric=method, uid=0, edge_probs=edge_probs,
                              real_ratio=(real_ratio[method], r))
    print("precompute fidelity")
    fid_precompute= [[(get_embs(x12, g, lay).detach(), x12.rule_evaluator.get_output(g).detach().softmax(dim=1)[0, 1])
                    for g in tqdm(gdset)] for lay in range(3)]
    print("precompute fidelity finished")

    outdf = pd.DataFrame(columns=["dataset", "metric", "graphid", "rule", "pred", "fidelity", "infidelity", "mags", "spars"])
    metric=method
    egos = dict()
    masks = dict()
    for rule in tqdm(rules):

        for x in [0]:#(range(nxp)):
            explainer = MCTSExplainer(model, (graphs, features, labels), 6, 10, 1, dataset, target_rule=rule,
                                                  target_metric=metric, uid=x, edge_probs=edge_probs,
                                                  real_ratio= (real_ratio[metric],r))
            #explainer.train(nsteps)
            if small :
                ratio =0.5
                file = "results/" + dir_name + "_small/dataset_" + dataset + "_rule_" + str(
                    rule) + "metric_" + metric + "xp_" + str(x) + "steps_" + str(nsteps) + "nxp_" + str(
                    nxp) + "ratio_" + str(ratio) + ".pkl"
                with open(file, "rb") as f1:
                    data = pickle.load(f1)
                    nodes = data
            else :
                file = "results/"+dir_name+"/dataset_" + dataset +"_rule_" + str(rule)+"metric_"+metric+"xp_" +str(x)+"steps_"+str(nsteps)+ "nxp_"+ str(nxp)+"ratio_"+str(0.5)+".pkl"
                #file = "results/"+dir_name+"/dataset_" + dataset +"_rule_" + str(rule)+"metric_"+metric+"xp_" +str(x)+"steps_"+str(nsteps)+ "nxp_"+ str(nxp)+".pkl"
                with open(file, 'rb') as f1:
                    data = pickle.load(f1)
                    #explainer.root= data
                nl = [el for el in data.as_list() if el.emb is not None]
                #best = max(nl, key= node.own_value)
                nodes = sorted(nl,key = lambda x :x.own_value)#explainer.rule_evaluator, metric)
            l = explainer.rule_evaluator.get_layer()+1
            nodes2 =list()
            while len(nodes2)==0:
                nodes2= [g for g in nodes if g.layer==l]
                l-=1

            #best_ego = [e.graph for e in nodes2[-1:]]
            best_ego = nodes2[-1].graph
            masks[rule] = explainer.rule_evaluator.get_output(best_ego).detach().softmax(dim=1)[0, 1]

            best_ego.nodes[0]["label"] = "center" + str(best_ego.nodes[0]["label"])

            egos[rule]=best_ego

    for i, (adj,f,(emb, pred), g) in tqdm(list(enumerate(zip(graphs,features,fid_precompute[explainer.rule_evaluator.get_layer()], gdset)))):
        #if i not in [134]:
        df = get_fidelity(explainer.rule_evaluator,metric,  dataset,i,emb, pred,adj,f, g, egos, masks)

        """outdf "= outdf.append([pd.Series([dataset, 0, metric,
                                         np.abs(preds[0]-preds[1]),
                                         np.abs(preds[0]-preds[2]),
                                         prop,
                                         spars], outdf.columns)])"""
        outdf = outdf.append(df)
        #xx = outdf.groupby("graphid").apply(get_best_fid_infid)
        outdf.to_csv("results/dataframes/egofidsinfids " + "_" + dataset +"_"+method+ ".csv")


    #print(fids)
    #print(sum(fids))
    print(infidelity)
    print("end")

def get_best_fid_infid(x):
    ratio =0.98
    fid = (x["pred"]-x["fidelity"]).abs()
    ff=  x[fid>=ratio*fid.max()]
    return ff.iloc[ff["infidelity"].argmin()]

def get_best_fid_sparse(x):
    ratio =0.95

    fid = (x["pred"]-x["fidelity"]).abs()
    ff=  x[fid>=ratio*fid.max()]
    return ff.iloc[ff["spars"].argmin()]


def get_best_polarised(x):
    fid = (x["pred"]-x["fidelity"]).abs()
    ff=  x[fid>=0.98*fid.max()]
    return ff.iloc[ff["infidelity"].argmin()]


def graph_dset(dset, atoms):
    graphs = list()

    for graph, features in zip(*dset):
        dense_adj = to_dense_adj(torch.tensor(graph)).numpy()[0]
        size = int(features.sum(axis=1).sum())
        g = nx.from_numpy_matrix(dense_adj[:size,:size])  # nx.Graph()
        g.add_nodes_from([(i, {"label": atoms[el]}) for i, el in enumerate(np.argmax(features[:size], axis=1))])

        graphs.append(g)
    return graphs

def get_embs(evaluator, graph, layer):

    X = evaluator.compute_feature_matrix(graph).type(torch.float32)
    A = torch.from_numpy(networkx.convert_matrix.to_numpy_array(graph))
    A = dense_to_sparse(A)[0]
    embeddinng = evaluator.gnnNets.embeddings(X, A)[layer]
    return embeddinng

def get_fidelity(evaluator, metric, dataset, id, embs,pred, adj, feat, graph, ego,masks, mode=None):

    rules = RuleEvaluator.load_rules(dataset)

    #masks = {k: [evaluator.get_output(e).detach().softmax(dim=1)[0, 1]for e in eg] for k, eg in ego.items() }

    outdf = pd.DataFrame(columns=["dataset", "metric", "graphid", "rule", "pred", "fidelity", "infidelity", "mags", "spars"])

    for rulenumber, egg in ego.items():
        target_rule = rules[rulenumber]

        acts = [RuleEvaluator.activate_static (target_rule, emb) for emb in embs]

        for egid, eg in enumerate([egg]):
            e= eg
            #e = eg.copy()
            #e.nodes[0]["label"] = "center" + e.nodes[0]["label"]
            remove_nodes = set()
            actsi = [i for i, el in enumerate(acts) if el]
            for i in random.sample(actsi, min(len(actsi), 10)):
                ii= get_intersiction(evaluator, i, graph, e, pred)
                remove_nodes = remove_nodes.union(ii)
            g2 = graph.copy()
            g2.remove_nodes_from(remove_nodes)
            if len(g2):
                pred2 = evaluator.get_output(g2).detach().softmax(dim=1)[0, 1]

                outdf = outdf.append([pd.Series([dataset, metric, id, rulenumber,
                                                 pred.item(),
                                                 pred2.item(),
                                                 masks[rulenumber].item(),
                                                 len(eg),
                                                 len(remove_nodes) / len(graph)], outdf.columns)])


    return outdf#maxp!=0, preds, prop, spars



def get_intersiction(evaluator, node, graph, ego,pred):
    gr = graph.copy()
    gr.nodes[node]["label"] = "center" + str(gr.nodes[node]["label"])
    iso = ISMAGS(gr, ego, node_match=lambda x, y: x["label"] == y["label"])

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(2)
    iso2 = list()
    # iso2 = [i for i in iso.largest_common_subgraph() if (node in i.keys())]
    maxf = -1
    try:
        for i in iso.largest_common_subgraph():
            if (node in i.keys()):
                iso2.append(i)
    except Exception:
        pass
    signal.alarm(0)
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(2)
    try:
        for i in iso2:
            g2 = graph.copy()
            g2.remove_nodes_from(i.keys())
            if len(g2):
                pred2 = evaluator.get_output(g2).detach().softmax(dim=1)[0, 1]
                if np.abs(pred-pred2)>maxf:
                    best_i = set(i.keys())
                    maxf = np.abs(pred-pred2)
    except Exception:
        pass
    signal.alarm(0)
    if maxf !=-1:
        return best_i

    #for i in iso:
    #    return set(i.keys())
    return {}


def read_results(dataset,method):
    f = "results/dataframes/egofidsinfids " + "_" + dataset +"_"+method+ ".csv"

    data = pd.read_csv(f)
    xx = data.groupby("graphid").apply(get_best_fid_infid)


    pol_01 = xx[xx["pred"]<0.5]
    pol_10 = xx[xx["pred"]>=0.5]
    print(dataset, method)
    print("base")
    print_fids_infids(xx)

    print("polarisé")
    fid_pol(xx)


    print(0)
    print("")

def fid_pol(df):
    pol_01 = df[df["pred"]<=0.5]
    pol_10 = df[df["pred"]>0.5]
    fid_01 = ((pol_01["pred"] > 0.5) != (pol_01["fidelity"] > 0.5)).mean()
    fid_10 = ((pol_10["pred"] > 0.5) != (pol_10["fidelity"] > 0.5)).mean()
    print("& " + format(fid_01, '.3f') + " & " + format(fid_10, '.3f'))


def print_fids_infids(df):
    fid_prob = (df["pred"] - df["fidelity"]).abs().mean()
    fid_acc = ((df["fidelity"]>0.5)==(df["pred"]<=0.5)).mean()
    infid_prob = (df["pred"] - df["infidelity"]).abs().mean()
    infid_acc= ((df["infidelity"]>0.5)==(df["pred"]<=0.5)).mean()

    print("& " +format(fid_prob,'.3f') + " & "+format(fid_acc,'.3f') )
    print("& " +format(infid_prob,'.3f') + " & "+format(infid_acc,'.3f') )
    print("& " +format(1-df["spars"].mean(),'.3f') )

    print("fidelity prob  : "+ str(fid_prob))
    print("fidelity acc  : "+ str(fid_acc))

    print("infidelity prob     : "+ str(infid_prob))
    print("infidelity acc     : "+ str(infid_acc))

    print("sparsity        : "+ str(1-df["spars"].mean()))
#aids","mutag","BBBP", "DD", "PROTEINS_full
#"aids","mutag","BBBP","DD" ]:#,
for dataset in ["PROTEINS_full"]:
    for method in ["cosine", "entropy", "likelyhood_max"]:
        run_masking(dataset, method, small=True)
        read_results(dataset, method)



#read_results("aids", "cosine")











def get_infidelity(evaluator, fid_precompute,graphs, ego):
    out = list()
    #ego =  [ego[-1]]

    sparsity = list()
    masks = [ evaluator.get_output(e).detach().softmax(dim=1)[0, 1]  for e in ego]
    spars = 0
    for (embs, pred), g in zip(fid_precompute, graphs):
        minp = 100

        #embs= get_embs(evaluator, g, evaluator.get_layer())
        if sum([evaluator.activate(emb.detach() )for emb in embs]):

            for e, eg in zip(masks, ego):
                if (np.abs(pred-e) < minp):
                    minp = np.abs(pred-e)

                    spars = len(eg) / len(g)

            out.append(minp)
            sparsity.append(spars)
    return out, sparsity




