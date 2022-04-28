import pickle as pkl



def repkl(name):
    with open("pkls/"+name+".pkl", 'rb') as fin:
        adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, edge_label_matrix= pkl.load(fin)

    out = adj, features, y_train+ y_val+y_test
    with open("pkls2/"+name+".pkl",'wb') as fout:
        pkl.dump(out,fout)

datasets= ["syn1","syn2","syn3","syn4",]

for d in datasets:
    repkl(d)