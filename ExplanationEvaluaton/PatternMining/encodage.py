import pandas as pd
import numpy as np
import csv
from optparse import OptionParser
from math import *
from numpy import linalg as LA

def lecture_motifs(nomRules, nomMotifs):
    df = pd.read_csv(nomRules, index_col = 0)
    idNum = df['id']
    idGraph = range(len(idNum))
    #print(idGraph, idNum)
    fichier = open(nomMotifs, "r")
    nbmotifs = 0
    data = fichier.readlines()
    for ligne in data:
        if(ligne[0] == '#'):
            nbmotifs = nbmotifs + 1
    #print("nb motifs",nbmotifs)
    fichier.close()
    
    x = 0
    fichier = open(nomMotifs, "r")
    res = np.zeros((len(df.values), nbmotifs))        
    data = fichier.readlines()
    numMotif = 0
    n1 = 0
    n2 = 0
    nbmotifsNonNuls = 0
    noms = []
    for ligne in data:
        if(ligne[0] == '#'):
            n1 = 0
            n2 = 0
            d = ligne.find('=')
            motif = ligne[d+1:len(ligne)-2]
            m = []
            if(len(motif) > 0):
                noms.append(ligne)
                nbmotifsNonNuls = nbmotifsNonNuls + 1
                k = motif.find(' ')    
                while(k>0):
                    att = motif[0:k]
                    m.append(att.strip())
                    motif = motif[k+1:len(motif)]
                    k = motif.find(' ')
                m.append(motif[0:len(motif)])
                    
                for i in range(len(df.values)):
                    supp = True
                    j = 0
                    while(supp and (j < len(m))):
                        if (df[m[j]].values[i] == 0.0):
                            supp = False
                        j = j + 1
                    if(supp == True):
                        res[i][nbmotifsNonNuls-1] = 1
                        x = x + 1
                        n2 = n2 + 1
            #else:
            #    print(ligne)
            #    print("motif vide")
            numMotif = numMotif + 1
        else:
            n1 = n1 + 1
    fichier.close()
    #print("x",x," : ",len(df.values),"x",nbmotifs,nbmotifsNonNuls)
    for i in range(len(idNum)):
        j = int(idNum[i])
        idNum[i] = int(idGraph[j])
    res = res[:,0:(len(res[0]) - (numMotif - nbmotifsNonNuls))]
    df1 = pd.DataFrame(data=res)
    frame=[idNum, df1, df['class']]
    result = pd.concat(frame, axis=1)
    r = result.groupby(by=["id"]).sum()
    #print(r)
    d = nomRules.find('.')
    nomout = nomRules[0:d]+"_encode.csv"
    r.to_csv(nomout, index=True)
    nomout1 = nomRules[0:d]+"_encode_motifs.csv"
    file = open(nomout1,"w")
    for i in range(len(noms)):
        file.write(str(i)+','+noms[i]+"")
    file.close()
    return r

   
if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-i", "--donneesact", dest="donneesActivations",
                      help="chemin du fichier d'instances", metavar="FILE")
    parser.add_option("-j", "--donneesmot", dest="donneesMotifs",
                      help="chemin du fichier d'instances", metavar="FILE")
    
    (options, args) = parser.parse_args()

    if ((not options.donneesActivations) or (not options.donneesMotifs)):
        parser.error('-i (--donnees) option required')
        parser.error('-j (--donnesMotifs) option required')
           
    else:
        nomrules = options.donneesActivations
        nommotifs = options.donneesMotifs
        res = lecture_motifs(nomrules, nommotifs)
        
