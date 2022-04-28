import pandas as pd
import numpy as np
import csv
from optparse import OptionParser
from math import *
from numpy import linalg as LA


def lecture_molecule(nom):
    dmol = pd.read_csv(nom, index_col = 0)
    data = dmol.values
    molecule = data[:,0]
    return molecule

def lecture_motifs(nomr, fw, molecule, nbm):
    print(nomr)
    fichier = open(nomr, "r")
    data = fichier.readlines()
    for ligne in data:
        if(ligne[0] == '#'):
            fw.write("# "+str(nbm)+ ligne[1:len(ligne)])
            nbm = nbm + 1
        else:
            if(ligne[0] == ' '):
                pass
            else:
                d = ligne.find(' ')
                som = int(ligne[0:d])
                e = ligne[d+1:len(ligne)-1]
                f = e.find(' ')
                mol = int(float(e[0:f]))
                fw.write(str(som) + " " + str(molecule[mol])+"\n")
                
    fichier.close()
    return nbm

   
if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-i", "--donnees", dest="donnees",
                      help="chemin du fichier d'instances", metavar="FILE")
    parser.add_option("-l", "--nblayers", dest="layers",
                      help="le layer", metavar="int")
      
    (options, args) = parser.parse_args()

    if ((not options.donnees) or (not options.layers)):
        parser.error('-i (--donnees) option required')
        parser.error('-l (--nblayersdonnees) option required')
             
    else:
        nom = options.donnees
        nblayers = options.layers
        d = nom.find('.')
        n = nom[0:d]
        mol = lecture_molecule(n+".csv")
        molecule = range(len(mol))
        nomw = n + "_motifs.txt"
        fw = open(nomw, "w")
        n = n + "_pretraite_"
        a = 0
        for i in range(int(nblayers)):
            for j in range(2):
                nl = n+str(i)+"_lay_"+str(i)+"_cib_"+str(j)+"res.txt"
                print(nl)
                a = lecture_motifs(nl, fw, molecule, a)
        fw.close()
        
