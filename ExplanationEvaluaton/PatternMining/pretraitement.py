import pandas as pd
import numpy as np
import csv
from optparse import OptionParser
from math import *
from numpy import linalg as LA
import os

def lecture_fichier(nom):
    print("File sorted by graph ids")
    print(os.system("pwd"))
    df = pd.read_csv(nom, index_col = 0)
    c = df.columns
    cols = c[1:len(c)-2]
    data = df.values
    molecule = data[:,0]
    classe = data[:,len(data[0]) - 1]
    data = data > 0
    data = data[:,1:len(data[0]) - 2]
    arr = np.sum(data, axis=1).argsort()
    data = data[arr[::]]

    molecule = molecule[arr]
    classe = classe[arr]
    sommet = np.array(range(len(data)))
    sommet = sommet[arr]
    
    p = len(data[0])
    n = len(data)
    ## layer selection
    indices_col = []
    ind = 0
    for i in range(p):
        garde = False
        name = cols[i]
        if(idLayer != -1):
            d = name.find('_') + 1
            f = name.find('c')
            layer = int(name[d:f])
        if((idLayer == -1) or (layer == idLayer)):
            garde = True
            indices_col.append(i)
    p = len(indices_col)
    val = np.zeros((n,p), dtype=bool)
    for i in range(n):
        for j in range(len(indices_col)):
            val[i][j] = data[i, indices_col[j]]
    data = val
    n = len(data)
    p = len(data[0])
    nb = 0
    val = np.zeros(n)
    for i in range(n):
        j = i+1
        while((j < n) and (val[j] == 0) and (val[i] == 0)):
            if(molecule[i] == molecule[j]):
                a = data[i,:]
                b = data[j,:]
                inclus = True
                k = 0
                while(inclus and (k < p)):
                    if((a[k] == True) and (b[k] == False)):
                        inclus = False
                    k = k + 1
                if(inclus == True):
                    val[i] = 1
                    nb = nb + 1
                    j = n
            j = j + 1
    new_n = n - nb
    b = np.zeros((new_n,p))
    mm = np.zeros(new_n)
    aa = np.zeros(new_n)
    cc = np.zeros(new_n)
    ind = 0
    ss = np.zeros(new_n)
    for i in range(n):
        if(val[i] == 0):
            for j in range(p):
                b[ind][j] = data[i][j]
            mm[ind] = molecule[i]
            cc[ind] = classe[i]
            ss[ind] = sommet[i]
            ind = ind + 1
    
    print(len(b),"x",len(b[0]))
    arr = mm.argsort()
    b = b[arr[::]]
    mm = mm[arr]
    cc = cc[arr]
    ss = ss[arr]
    nom = nom[0:nom.find(".")]
    nom1 = nom +"_pretraite_"+str(idLayer)+".csv"
    f_data = open(nom1, "w")
    f_data.write(',id,')
    for i in range(len(indices_col)-1):
        f_data.write(cols[indices_col[i]]+',')
    f_data.write(cols[indices_col[len(indices_col)-1]]+",sommet,class\n")
    
    for i in range(len(b)):
        f_data.write(str(i)+','+str(mm[i])+","),
        for j in range(len(b[i])):
            f_data.write(str(b[i][j])+',')
        
        f_data.write(str(ss[i])+','+str(cc[i])+"\n"),
        
    f_data.close()
    


if __name__ == "__main__":

    
    parser = OptionParser()
    parser.add_option("-i", "--donnees", dest="donnees",
                      help="chemin du fichier d'instances", metavar="FILE")
    parser.add_option("-l", "--layer", dest="layer",
                      help="le layer", metavar="int")
 
    (options, args) = parser.parse_args()

    if ((not options.donnees) or (not options.layer)):
        parser.error('-i (--donnees) option required')
        parser.error('-l (--layer) option required')
        
    else:
        nom = options.donnees
        idLayer = int(options.layer)
        lecture_fichier(nom)
 
