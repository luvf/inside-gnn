
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

import matplotlib.pyplot as plt
import seaborn as sns

Number_of_rules = dict([("aids", 60), ('mutag', 60), ("BBBP", 60), ("PROTEINS_full", 28), ("DD", 47), ("ba2", 19)])


def get_children_values(tree, fun):
    nodes = tree.as_list()
    nodes = [el for el in nodes if el.emb is not None]

    vals = list()
    for node in nodes:
        if node.emb is not None:
            v1 = fun(node)
            for el in node.children:
                if el.emb is not None:
                    vals.append(np.abs(v1-fun(el)))
    return np.mean(vals)

def get_distribution_values(tree,evaluator, metr, rules, rule):
    nodes = tree.as_list()

    nodelist =  [el for el in nodes if el.emb is not None]
    best_nodes = sorted(nodelist, key = (lambda el: evaluator.funcs[metr](el.emb, evaluator.rules[rule])))[-10:]

    vals =list()

    for r in rules:
        rulenb = evaluator.rules[r]
        rr= list()
        for node in best_nodes:
            vv = evaluator.funcs[metr](node.emb, rulenb)
            rr.append(vv)
        vals.append((np.mean(rr),np.std(rr)))
    return vals

def get_distribution_realism(tree,evaluator, metr, rules, rule):
    nodes = tree.as_list()

    nodelist = [el for el in nodes if el.emb is not None]
    #top = sorted(nodelist, key=(lambda el: evaluator.funcs[metr](el.emb, evaluator.rules[rule])))[-20:]
    top = sorted(nodelist, key=(lambda el: el.value/el.visit))[-100:]

    #evaluator.funcs[metr](node.emb, rulenb)
    top_vals = list(map(lambda el: evaluator.funcs[metr](el.emb, evaluator.rules[rule]), top))
    top_real = list(map(lambda el:evaluator.real_score(el.graph), top))


    values = list(map(lambda el: evaluator.funcs[metr](el.emb, evaluator.rules[rule]), nodelist))
    real = list(map(lambda el:evaluator.real_score(el.graph), nodelist))


    #values = sorted(values, )
    #sns.displot(list(values),kind="kde", color='blue')


    df=pd.DataFrame(list(zip(values, real, ["all" for _ in values]))+ list(zip(top_vals, top_real, ["top" for _  in top_vals ])), columns=["score", "realism","type"])
    #d2= pd.DataFrame(), columns=["score", "realism","type"])

    #xx = plt.subplots(1)
    plt.figure(figsize=(20, 20))

    #g = sns.JointGrid(data=df, x="score", y="realism",hue="type")
    #g.plot(sns.kdeplot, sns.histplot,kde=True, stat="count", bins=20)

    g = sns.jointplot(data=df, x="score", y="realism",hue="type")
    g.plot_joint(sns.kdeplot, zorder=0, levels=20)
    g.plot_marginals(sns.histplot,kde=True, clip_on=False)

    g.figure.suptitle("rule " + str(rule) +" metric "+ metr +" elements " + str(len(nodelist)))

    #sns.jointplot(data=df, x="score", y="realism",hue="type")

    #g.show()
    #plt.savefig("results/plts/rule" + str(rule) +"_metric_"+ metr +".png",format='png')
    plt.show()


def get_mcts_vals(dataset):
    graphs, features, labels, _, _, _ = load_dataset(dataset)

    if dataset =="ba2":
        edge_probs=None
    else :
        if dataset =="PROTEINS_full":
            edge_probs = get_edge_distribution(graphs, features,30)
        else :
            edge_probs = get_edge_distribution(graphs, features)
        #degre_distrib = get_degre_distribution(graphs, features)

    model, checkpoint = model_selector("GNN",
                                       dataset,
                                       pretrained=True,
                                       return_checkpoint=True)
    metrics = ["cosine","likelyhood_max", "entropy"]
    #metrics = []

    real_ratio = {"cosine": (0, 1),
                  "entropy" : (-70,-2.014),
                  "lin" : (-1,0.1),
                  "likelyhood_max": (-35,20.04)
                  }
    #metrics = ["likelyhood_max"]
    eval_metrics = []#metrics[0]
    #metrics = ["likelyhood"]
    rules = range(Number_of_rules[dataset])

    rules = [list(range(0,20)),list(range(20,40)), list(range(40,60))]



    scores = list()
    nsteps = 2000
    nxp = 1
    print( dataset+ " "+ str(nsteps) + " " + str(nxp)+ " " + str(metrics))
    r=1
    vals = list()
    for rulelist in rules:
        for rule in tqdm(rulelist):

            for metric in metrics:
                for x in (range(nxp)):
                    explainer = MCTSExplainer(model, (graphs, features, labels), 6, 10, 1, dataset, target_rule=rule,
                                                  target_metric=metric, eval_metrics=[metric], uid=x, edge_probs=edge_probs,
                                                  real_ratio= (real_ratio[metric],r))
                    #explainer.train(nsteps)
                    #rv = partial(rule_value, explainer.rule_evaluator, metric, rule)
                    #dd= get_distribution_values(explainer.tree.root,rv )
                    #valtree = get_children_values(explainer.tree.root, rv)
                    #vals = [get_distribution_values(explainer.tree.root,partial(rule_value, explainer.rule_evaluator, metric, rn)) for rn in tqdm(range(20*(rule//60), 20*(rule//60)+20))]
                    #with open("results/mcts_dumps/dataset_" +"valtree dataset +"_rule_" + str(rule)+"metric_"+metric+"xp_" +str(x)+"steps_"+str(nsteps)+ "nxp_"+ str(nxp)+".pkl", 'wb') as f1:
                    #    explainer.tree.root.clean_tree()

                    #    pickle.dump(explainer.tree.root, f1)
                    dir = "mcts_dumps_old"
                    old_nodes = get_nodes("results/"+dir+"/dataset_" + dataset +"_rule_" + str(rule)+"metric_"+metric+"xp_" +str(x)+"steps_"+str(nsteps)+ "nxp_"+ str(1)+".pkl",
                              explainer)
                    dir= "mcts_dumps_new"
                    new_nodes = get_nodes("results/" + dir + "/dataset_" + dataset + "_rule_" + str(
                        rule) + "metric_" + metric + "xp_" + str(x) + "steps_" + str(nsteps) + "nxp_" + str(1) + ".pkl",
                                          explainer)
                    get_dis(old_nodes, new_nodes, rule, metric)
                    """with open("results/"+dir+"/dataset_" + dataset +"_rule_" + str(rule)+"metric_"+metric+"xp_" +str(x)+"steps_"+str(nsteps)+ "nxp_"+ str(1)+".pkl", 'rb') as f:
                        base = pickle.load(f)
                        
                        x=tt
                        l_node=[el for el in tt.as_list() if el.emb is not None]
                        scores=[explainer.compute_score(el.graph)[1][0] for el in l_node]
                        sns.displot(scores)
                        plt.show()
                        #vv = get_distribution_values(tt ,explainer.rule_evaluator, metric, rulelist,rule)
                        #get_distribution_realism(tt ,explainer.rule_evaluator, metric, rulelist,rule)
                        #vals.append(get_distribution_values(tt ,explainer.rule_evaluator, metric, rulelist,rule))
                    #with open("results/mcts_dumps/dataset_" + dataset +"metric_"+metric+"confusion.pkl", 'wb') as f:
                    #    pickle.dump(np.array(vals), f)"""
    print(0)

def get_dis(old,new,rule, metr):
    old = sorted(old, key=lambda x :x[0])[-100:]
    olds = [el[1][0] for el in old]
    oldr = [el[1][1] for el in old]

    new = sorted(new, key=lambda x :x[0])[-100:]

    news = [el[1][0] for el in new]
    newr = [el[1][1] for el in new]

    df=pd.DataFrame(list(zip(olds, oldr,  ["old" for _ in oldr]))+ list(zip(news,newr, ["new" for _ in newr])), columns=["score","realism","type"])
    g = sns.jointplot(data=df, x="score", y="realism",hue="type")
    g.plot_joint(sns.kdeplot, zorder=0, levels=20)
    g.plot_marginals(sns.histplot,kde=True, clip_on=False)

    g.figure.suptitle("rule " + str(rule) +" metric "+ metr )
    plt.show()

def get_nodes(filename, explainer):
    with open(filename, 'rb') as f:
        tree = pickle.load(f)
        l_node = [el for el in tree.as_list() if (el.emb is not None or len(el.graph)>1)]
        scores = [explainer.compute_score(el.graph) for el in l_node]
        return scores
#for dataset in ["aids"]:#', "","ba2"]:

#    get_mcts_vals(dataset)



"""
En gros pour les figures 2, 3, 4 et 5 on a besoin en ligne des patterns et en colonne 1) la méthode, 2) le numéro de layer, 3) le nombre de "supporting nodes", 4) le nombre de composantes activées, 5) SI_SG, 6) coverage positive class, 7) coverage negative class.
11:37
je ne sais pas si les informations nécessaires à la figure 7 peuvent être ajoutées au fichier précédent ou pas, si oui en colonne à la suite
"""

from ExplanationEvaluation.explainers.utils import RuleEvaluator
Number_of_rules = dict([("aids", 60), ('mutag', 60), ("BBBP", 60), ("PROTEINS_full", 28), ("DD", 47), ("ba2", 19)])
from ExplanationEvaluation.gspan_mine.visualize import Report


def statistical_data(dataset):
    names = {"ba2": ("ba2"),
             "aids": ("Aids"),
             "BBBP": ("Bbbp"),
             "mutag": ("Mutag"),
             "DD": ("DD"),
             "PROTEINS_full": ("Proteins")
             }
    name = names[dataset]
    file = "ExplanationEvaluation/datasets/activations/" + name + "/" + name + "_activation_encode_motifs.csv"
    # file = "/home/ata/ENS de Lyon/Internship/Projects/MCTS/inter-compres/INSIDE-GNN/data/Aids/Aids_activation_encode_motifs.csv"
    rules = list()
    datas = load_dataset(dataset)[:3]

    gspan_report =  Report("results/result_egos2.json")
    model, checkpoint = model_selector("GNN",
                                       dataset,
                                       pretrained=True,
                                       return_checkpoint=True)
    embs = get_embs(datas[:2], model)
    with open(file, "r") as f:
        for l in f:
            r = l.split("=")[1].split(" \n")[0]
            dd=dict()
            for el in l.split(" "):
                s = el.split(":")
                if len(s) ==2:
                    dd[s[0]] = float(s[1])


            label = int(l.split(" ")[3].split(":")[1])
            if Number_of_rules[dataset]>len(rules):
                rules.append((label, r, dd))
    out = list()

    label_prop = datas[2].sum(axis=0)
    for i, (_, r, dd) in enumerate(tqdm(rules)):
        c = r.split(" ")
        layer = int(c[0][2])
        components = list()
        with open("results/gspan/gspan_"+dataset+"_rule_"+str(i)+".pkl", "rb") as f:
            xx = pickle.load(f)
        wracc2 = xx["value"]
        for el in c:
            components.append(int(el[5:]))
            #(layer, components, components, layer)
        acts = count_activation(embs, (layer, components))
        target = int(dd["target"])
        act_graphs = [int(dd["c-"]), int(dd["c+"])]

        #wracc = sum(act_graphs)/label_prop.sum()*(act_graphs[target]/sum(act_graphs) - label_prop[target]/label_prop.sum())
        wracc = [ report["WRAcc"] for report in gspan_report.reports if report_helper(report, dataset, i )]
        if len(wracc)==0:
            wracc =[-1]
        out.append((i, layer, acts, int(dd["nb"]) , format(dd["score"] ,'.3f'), int(dd["c+"]), int(dd["c-"]), int(dd["target"]), sum(wracc)/len(wracc), wracc2))
    return out

def report_helper(report, dataset, rule):
    return report["dataset_name"]== dataset+"_"+str(rule)+"labels_egos"

def get_embs(dataset, model):
    embs = list()
    for g, f in tqdm(list(zip(*dataset))):
        g = torch.tensor(g)
        f = torch.tensor(f)
        #adj = dense_to_sparse(g)[0]
        max = int(f.sum())
        embs.append([el[:max] for el in model.embeddings(f, g)])
    return embs

def count_activation(embs, rule):
    count=0

    for e in embs:
        for node in e[rule[0]]:
            count += RuleEvaluator.activate_static(rule, node)
    return count



""" des patterns et en colonne 1) la méthode, 2) le numéro de layer, 3) le nombre de "supporting nodes", 4) le nombre de composantes activées, 5) SI_SG, 6) coverage positive class, 7) coverage negative class.
"""
"""
print("looo")
import csv
#aids", "mutag",  "BBBP","ba2", "DD" , "PROTEINS_full
for dataset in [ "aids", "mutag",  "BBBP"]:
    data = statistical_data(dataset)

    with open("results/stat_data_"+dataset+".csv", 'w', newline='') as csvfile:
        datawriter = csv.writer(csvfile)
        header = ['pattern number', 'layer', 'supporting nodes', "activated components","SI_SG", "pos graphs", "neg graphs", "target class ","wracc","w2"]
        datawriter.writerow(header)
        for row in data :
            datawriter.writerow(row)


"""

base_rule =[0,5]
indexes=[1,2, 4,5]
components = [0, 2, 7, 8, 10, 16]


def latex_embs(data, components,  rule, fig_id, color=True ):
    out = str()


    for i, vl in enumerate(data):
        if i== 0:
            c=" \\multirow{"+str(len(data))+"}{5em}{\\includegraphics[scale=0.38]{mols/mol"+str(fig_id+1)+".pdf}} & \\multirow{"+str(len(data))+"}{1em}{$c_0$}&"
        else :
            c=" & & "
        if all(vl[components][rule]):
            c+= "\\cellcolor{red}"
        l = c + "$" + str(i + 1) + "$"

        for v in vl[components]:
            vv = format(v ,'.1f')
            l+= " & "
            if float(vv)>= 0.1 and color:
                l+= "\\cellcolor[gray]{0.8}" # COLOR
            l+= "$" +vv + "$"

        if i == len(data)-1:
            l += "\\\\\\hline\n"
        else :
            l += "\\\\ \hhline{|*{2}{~|}*{" + str(len(components) + 1) + "}{-}}\n"
        out+= l[1:]

    print(out)


def latex_node_indices(nlines):
    out = str()
    l0 ="\\begin{tabular}{|>{\centering\\arraybackslash}p{"+size+"}|}\\hline\n"
    lend = "\\end{tabular}"
    for i in range(nlines):
        l = str()



        l+="$"+str(i+1)+"$\\\\\\hline\n"
        out+= l
    print(l0+ out+lend)


mols = ["HOCNC", "HCCNC", "CHCCN", "CCCNO", "HOCCC", "HCCNCO"]

adjs = [[
    [1,0,0,1,0],
    [0,1,0,0,1],
    [0,0,1,1,0],
    [1,0,1,1,1],
    [0,1,0,1,1],
],[
    [1,0,0,1,0],
    [0,1,0,0,1],
    [0,0,1,1,0],
    [1,0,1,1,1],
    [0,1,0,1,1],
],[
    [1,0,0,1,0],
    [0,1,0,0,1],
    [0,0,1,1,0],
    [1,0,1,1,1],
    [0,1,0,1,1],
],[
    [1,0,0,1,0],
    [0,1,0,0,1],
    [0,0,1,1,0],
    [1,0,1,1,1],
    [0,1,0,1,1],
],[
    [1,0,0,1,0],
    [0,1,0,0,1],
    [0,0,1,1,0],
    [1,0,1,1,1],
    [0,1,0,1,1],
],
    [
        [1,0,0,1,0,0],
        [0,1,0,0,1,0],
        [0,0,1,1,0,1],
        [1,0,1,1,1,0],
        [0,1,0,1,1,0],
        [0,0,1,0,0,0],
    ],
]


from torch.nn.functional import one_hot, softmax

def convert(model, feat, adj, dataset):
    embs = list()

    atoms = get_atoms(dataset, None)
    revatoms = {v: k for k, v in atoms.items()}
    features = [revatoms[el] for el in feat]
    features= one_hot(torch.tensor(features),len(atoms)).type(torch.float32)
    adjs = torch.tensor(adj)
    adjs = dense_to_sparse(adjs)[0]

    return model.embeddings(features, adjs)





def exportlatex(dataset):
    model, checkpoint = model_selector("GNN",
                                       dataset,
                                       pretrained=True,
                                       return_checkpoint=True)
    print("\\begin{figure}")

    layers = [2]
    nb_components= len(components)
    size = "1.2em"

    l0 ="\\begin{tabular}{|c|c|c||"+ "".join([">{\centering}p{"+size+"}|" for _ in range(nb_components-1)]+[">{\centering\\arraybackslash}p{"+size+"}|"]) +"}\\hline\n"
    l0+="\\rotatebox{90}{graph} & \\rotatebox{90}{output class}& \\rotatebox{90}{node index}&" + "&".join(["\\rotatebox{90}{component "+str(i+1)+"}" for i in range(len(components))]) +"\\\\ \hline\n"
    print(l0)
    indexes=[1,2, 4,5]
    #molsxx = mols
    molsi = [mols[i] for i in indexes]
    adjsi = [adjs[i] for i in indexes]

    for ii, (feat, adj)  in enumerate(zip(molsi, adjsi)):
        emb = convert(model, feat, adj, dataset)
        #latex_node_indices(emb[0].shape[0])
        for l in layers:
            latex_embs(emb[l],components,base_rule,ii)
            print("\n\hline\hline")
    print("\end{tabular}")
    print("\\end{figure}")




def export_mcts_tree(dataset):
    model, checkpoint = model_selector("GNN",
                                       dataset,
                                       pretrained=True,
                                       return_checkpoint=True)
    graphs, features, labels, _, _, _ = load_dataset(dataset)
    edge_probs = get_edge_distribution(graphs, features,labels)
    metric = "cosine"
    real_ratio = (0, 1)
    r = 0.5
    nsteps = 1000

    new_rule = (2, [components[r] for r in base_rule])
    explainer = MCTSExplainer(model, (graphs, features, labels), 6, 10, 1, dataset, target_rule=0,
                              target_metric=metric, uid=-1, edge_probs=edge_probs,
                              real_ratio= (real_ratio,r))
    explainer.rule_evaluator.target_rule = new_rule

    explainer.train(nsteps)

    tree_list = explainer.tree.root.as_list()
    #tree_list = list(enumerate(tree_list))
    tree_list= [explainer.tree.root]+[el for el in explainer.tree.root.as_list() if el.own_value_detailed is not None and el.graph.nodes[0]["label"]in ["C","H","O","N"]]
    for i, el in enumerate(tree_list):
        for ch in el.children:
            ch.parent = i
    #[[(ch.parent=i) for ch in el.children] for i, el in enumerate(tree_list)]
    max_depth = [index for index, el in enumerate(tree_list) if el.layer==2]
    max_depth = random.sample(max_depth, 10)
    keep = np.zeros(len(tree_list))
    keep[0]=1

    for i in max_depth:
        keep[i]=1
        cur= tree_list[i]
        while cur.parent!=0:
            keep[cur.parent]=1
            cur=tree_list[cur.parent]
    filtred_mols = [(i,el) for i,(k, el) in enumerate(zip(keep, tree_list)) if k ]
    [latex_node(el, i , base_rule ) for i, el in filtred_mols ]

    save_mols(filtred_mols)
    print("lolo")


def plot_mol(molecule):
    ret=plt.figure(figsize=(4,4))
    #for i, el in enumerate(molecules):
    mol = molecule#to_networkx(molecule,node_attrs="x").to_undirected()
    labels= {node: molecule.nodes[node]["label"] for i,node in enumerate(molecule.nodes())}
    nodes = {node for node in molecule.nodes()}
    #plt.subplot(size[0],size[1], i+1)
    nx.draw_kamada_kawai(mol, node_size=1, with_labels=False)

    nx.draw_kamada_kawai(mol, nodelist=nodes, labels=labels, with_labels=True)
    if len(nodes):
        nx.draw_kamada_kawai(mol, nodelist=[0], node_color="r")

    return ret

def save_mols(nodes):
    for i, el in nodes:
        plot_mol(el.graph)
        plt.savefig("../../tex/tables/mols2/mol_"+str(i)+".png")

new_line_tab = "\\\\\\hhline{|*{1}{~}*{2}{-}}\n"

def latex_node(node,i=0, rule=[]):
    if node.emb is None:
        node.own_value_detailed = [0, 0]
    out = "\\begin{tabular}{|c|c|l|}\hline \n"
    out += " \\multirow{5}{5em}{\\includegraphics[scale=0.2]{/home/luca/codes/inter-compres/tex/tables/mols2/mol_" + str(i) + ".png}}"

    #out += "\\\\\\hline"

    out += "&score &  "+ format(node.own_value_detailed[0] ,'.2f') + new_line_tab

    out += "&realism &  "+ format(node.own_value_detailed[1] ,'.2f')+ new_line_tab

    out += "&value &  "+ format(node.value  ,'.2f')+ new_line_tab
    out += "&visit &  "+ str(node.visit)+ "\\\\\hline"


    if node.emb is not None:
        out += " \multicolumn{3}{|c|}{\cellcolor[gray]{0.8}"
        out += vector_sub_table(node.emb, rule)
        out += "}\n\\\\\hline"

    out += "\end{tabular}\n"
    out += "% ref : "+ str(i)+"\n % parent : " + str(node.parent)+" \n\n"

    print(out)

def vector_sub_table( emb, rule=list):
    size ="1.5em"
    out = "\\begin{tabular}{"+"|".join([">{\centering}p{"+size+"}" for _ in range(len(components)) ])+"}"
    color = lambda x: "\cellcolor{white}" if x not in rule else ""
    out += "&".join(color(i)+ format(emb[v] ,'.2f') for i,v in enumerate(components))
    out += "\end{tabular}\n\n"
    return out
export_mcts_tree("mutag")
#exportlatex("mutag")
