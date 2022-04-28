import pandas as pd
import numpy as np
import csv
from optparse import OptionParser
from math import *
from numpy import linalg as LA

SGset = []
pile_pattern = []
topk = None
minsupp = 0
nbenum = 0
prune = 0
weight0 = 0.0
weigth1 = 0.0
dico_atome={}
cible = 1
inf =10**6
beta = 1
alpha = 0.6 
        
def truncate(n):
    return int(n * 1000) / 1000

class Model:
    def __init__(self, pl, pc):
        global ds
        self.proba = []
        self.colM = pc
        self.ligneM = pl
        self.nunique = 0
        self.punique = 0
               
        self.mult_ligne = {}
        self.mult_col = {}
        self.cardmult_ligne = []
        self.cardmult_col = []
        self.keys_col = []
        self.keys_ligne = []
        
        self.lambdasT = []
        self.musT = []
        self.lambdas = np.zeros(ds.p)
        self.mus = np.zeros(ds.n)
       
        self.gradientCol = []
        self.gradientLigne = []
        
        self.hessianCol = []
        self.hessianLigne = []

    def maj(self,P):
        global cible
        for i in P.P:
            li = P.S + P.SYes
            for s in li:
                if(s.classe == cible):
                    self.proba[s.index][i.index] = 1.0
                
    def maj2(self,P):
        global ds
        global TrueLigne 
        global TrueCol 
       
        for i in P.P:
            v = int(round((len(P.S)/TrueCol[i.index]) * len(P.S)))
            self.colM[i.index] = self.colM[i.index] 
        for i in P.S:
            v = int(round((len(P.P)/TrueLigne[i.index]) * len(P.P)))
            self.ligneM[i.index] = self.ligneM[i.index] 
        ds.pligne = self.ligneM / np.sum(self.ligneM) 
        ds.pcol = self.colM / np.sum(self.ligneM) 
        #print("new pligne et pcol")
        #print(self.ligneM)
        #print(self.colM)
        self.compute_model()
        self.maj(P)
          
    def compute_multiplicities(self):
        global ds
        initval = -1
        for i in range(len(ds.pligne)):
            if ds.pligne[i] not in self.mult_ligne:
                self.mult_ligne[ds.pligne[i]] = [i]
                self.musT.append(initval)
            else:
                 self.mult_ligne[ds.pligne[i]].append(i)          
        for i in range(len(ds.pcol)):
            if ds.pcol[i] not in self.mult_col:
                self.mult_col[ds.pcol[i]] = [i]
                self.lambdasT.append(initval)
            else:
                 self.mult_col[ds.pcol[i]].append(i)
        
        self.keys_col = list(self.mult_col.keys())
        self.keys_ligne = list(self.mult_ligne.keys())
        self.keys_ligne = np.array(self.keys_ligne)
        self.keys_col = np.array(self.keys_col)
        
        for m in self.keys_col:
            self.cardmult_col.append(len(self.mult_col[m]))
        for m in self.keys_ligne:
            self.cardmult_ligne.append(len(self.mult_ligne[m]))
        self.cardmult_ligne = np.array(self.cardmult_ligne)
        self.cardmult_ligne = self.cardmult_ligne.reshape((1,len(self.musT)))
        self.cardmult_col = np.array(self.cardmult_col)
        self.cardmult_col = self.cardmult_col.reshape((1,len(self.lambdasT)))
        self.punique = len(self.lambdasT)
        self.nunique = len(self.musT)
        self.lambdasT = np.array(self.lambdasT).reshape((len(self.lambdasT),1))
        self.musT = np.array(self.musT).reshape((len(self.musT),1))
        
    def calcul_E(self, la, mu):
        e = np.exp(la).transpose().reshape((1,self.punique))
        ee = np.exp(mu).reshape((self.nunique,1))
        E = np.multiply(np.ones((self.nunique,1))*e, ee*np.ones((1,self.punique)))
        return E
    
    def calcul_grad(self, ps):
        global ds
        gla = np.multiply(-len(ds.pligne),self.keys_col.reshape((1,self.punique)))
        xx = np.dot(ps.transpose(), self.cardmult_ligne.transpose())
        gla =  gla.transpose() + xx           
        gmu = np.multiply(-len(ds.pcol),self.keys_ligne.reshape((1,self.nunique))) 
        xx = np.dot(ps, self.cardmult_col.transpose())
        gmu =  gmu.transpose() + xx
        #norm
        u = np.repeat(gla,self.cardmult_col[0,:])
        v = np.repeat(gmu,self.cardmult_ligne[0,:])
        n1 = LA.norm(u)
        n2 = LA.norm(v)
        error = LA.norm([n1,n2])
       
        return gla, gmu, error

    def calcul_hess(self, ps2):
        hla = np.dot(ps2.transpose(), self.cardmult_ligne.transpose())
        hmu = np.dot(ps2, self.cardmult_col.transpose())
        return hla, hmu
    
    def calcul_lagragian(self, la, mu, f, deltala, deltamu):
        u = la + f * deltala
        v = mu + f * deltamu
        return u,v
    
    def update(self,error):
        #global eps
        munique = len(self.lambdasT)
        nunique = len(self.musT)
        lambdasTry = np.ones(munique)
        musTry = np.ones(nunique)
        glambdasTry = np.ones(munique)
        gmusTry = np.ones(nunique)
        fbest = 0
        errorbest = error
        if 0 in self.hessianCol:
            deltamu = 0
            for i in range(len(self.hessianCol)):
                if(self.hessianCol[i] != 0):
                    deltamu = deltamu - (self.gradientCol[i] / self.hessianCol[i])
        else:
            deltamu = - (self.gradientCol / self.hessianCol)
        if 0 in self.hessianLigne:
            deltala = 0
            for i in range(len(self.hessianLigne)):
                if(self.hessianLigne[i] != 0):
                    deltala = deltala - (self.gradientLigne[i] / self.hessianLigne[i])
        else:
            deltala = - (self.gradientLigne / self.hessianLigne)
        
        for f in np.logspace(-5,1,100): 
            lambdasTry, musTry = self.calcul_lagragian(self.lambdasT, self.musT, f, deltala, deltamu)
            Etry = self.calcul_E(lambdasTry, musTry)
            if (Etry.max() < 1):
                pstry = np.divide(Etry,(1+Etry))
                glambdasTry, gmusTry, errorTry = self.calcul_grad(pstry)
                if (errorTry < errorbest):
                    fbest = f
                    errorbest = errorTry
        ## calcul lambda final
        self.lambdasT, self.musT = self.calcul_lagragian(self.lambdasT, self.musT, fbest, deltala, deltamu)
        return errorbest        
            
    def compute_model(self, nit = 100):
        global ds
        self.compute_multiplicities()
        for k in range(nit):
            #print("iteration ",k)
            #print("best", self.lambdasT, self.musT)
            E = self.calcul_E(self.lambdasT, self.musT)
            ps = E /(1+E)
            ps2 = np.divide(E,(1 + E)**2)
            self.gradientLigne, self.gradientCol, erreur = self.calcul_grad(ps)
            self.hessianLigne,  self.hessianCol = self.calcul_hess(ps2)
            err = self.update(erreur)
    
        E = self.calcul_E(self.lambdasT, self.musT)
        ps = np.divide(E,(1 + E))
        ps2 = np.divide(E,(1 + E)**2)
        self.gradientLigne, self.gradientCol, erreur = self.calcul_grad(ps)
        self.proba = np.zeros((ds.n, ds.p))
        i = 0
        for ki in self.mult_ligne:
            j  = 0
            for kj in self.mult_col:
                for a in self.mult_ligne[ki]:
                    for b in self.mult_col[kj]:
                        self.proba[a][b] = E[i][j]
                j  = j + 1
            i = i + 1    

class Atome:
    def __init__(self, index, classe, atome, molecule):
        self.index = index
        self.classe = classe
        self.atome = atome
        self.molecule = molecule

class Item:
    def __init__(self, index, layer, component, name):
        self.index = index
        self.layer = layer
        self.component = component
        self.name = name
        
class Pattern:
    def __init__(self, P, S, Not, Pot,SNot):
        self.P = P
        self.Not = Not
        self.Pot = Pot
        #on met les atomes
        self.S = S
        self.SNot = SNot
        self.SYes = []
        self.supp = 0
        self.nbClasse0 = 0
        self.nbClasse1 = 0
        self.SIP = 0.0
        self.SIPUB = 0.0
        self.SIM = 0.0
        self.SIMUB = 0.0
        self.SINegM = 0.0
        self.SINegP = 0.0
        self.DLP = 0.0
        self.DLPUB = 0.0
        self.DLM = 0.0
        self.DLMUB = 0.0
        self.IR = 0.0
        self.signe = 0
        self.nbMol = 0
        
        
def alog(x):
    if (x == 0):
        return 0.0
    else:
        return log(x)

def amin(x,y):
    if(x < y):
        return x
    else:
        return y

def amax(x,y):
    if(x > y):
        return x
    else:
        return y
        
class Dataset:
    def __init__(self, data, pcol, pligne):
        self.data = data
        self.n = len(data)
        self.p = len(data[0])
        self.pcol = pcol
        self.pligne = pligne
        self.listMolecules = {}
        self.pClasse1 = 0
        self.prob = 0.0


    def calcul_si(self, P):
        global weight1
        global weight0
        
        dico_maxP = {}
        dico_maxM = {}
        li = P.S + P.SYes
        for s in li: 
            mp = 0
            mm = 0
            for j in P.P:
                if(s.classe == 1):
                    mp = mp - weight1 * alog(model.proba[s.index][j.index])
                else:
                    mm = mm - weight0 * alog(model.proba[s.index][j.index])
                
            mol = dico_atome[s.index]
            if(mol in dico_maxP):
                if(mp > dico_maxP[mol]):
                    dico_maxP[mol] = mp
            else:
                dico_maxP[mol] = mp
            if(mol in dico_maxM):
                if(mm > dico_maxM[mol]):
                    dico_maxM[mol] = mm
            else:
                dico_maxM[mol] = mm
        x = 0
        y = 0
        for key in dico_maxP:
            x = x + dico_maxP[key]
        for key in dico_maxM:
            y = y + dico_maxM[key]
        del dico_maxP
        del dico_maxM
        return x, y

    def calcul_sietub(self, P):
        global weight1
        global weight0
        dico_maxP = {}
        dico_maxM = {}
        dico_maxPsi = {}
        dico_maxMsi = {}
        dico_maxNegP = {}
        dico_maxNegM = {}
        li = P.S + P.SYes
        for s in li:
            mp = 0
            mm = 0
            mpsi = 0
            mmsi = 0
            for j in P.P:
                if(s.classe == 1):
                    mp = mp - weight1 * alog(model.proba[s.index][j.index])
                else:
                    mm = mm - weight0 * alog(model.proba[s.index][j.index])
            mpsi = mp
            mmsi = mm
            for j in P.Pot:
                if(s.classe == 1):
                    mpsi = mpsi - weight1 * alog(model.proba[s.index][j.index])
                else:
                    mmsi = mmsi - weight0 * alog(model.proba[s.index][j.index])
                
            mol = dico_atome[s.index]
            if(mol in dico_maxP):
                if(mp > dico_maxP[mol]):
                    dico_maxP[mol] = mp
            else:
                dico_maxP[mol] = mp
            if(mol in dico_maxM):
                if(mm > dico_maxM[mol]):
                    dico_maxM[mol] = mm
            else:
                dico_maxM[mol] = mm
            if(mol in dico_maxPsi):
                if(mpsi > dico_maxPsi[mol]):
                    dico_maxPsi[mol] = mpsi
            else:
                dico_maxPsi[mol] = mpsi
            if(mol in dico_maxMsi):
                if(mmsi > dico_maxMsi[mol]):
                    dico_maxMsi[mol] = mmsi
            else:
                dico_maxMsi[mol] = mmsi
                
        for s in P.SYes:
            mp = 0
            mm = 0
            for j in P.P:
                if(s.classe == 1):
                    mp = mp - weight1 * alog(model.proba[s.index][j.index])
                else:
                    mm = mm - weight0 * alog(model.proba[s.index][j.index])
            mpsi = mp
            mmsi = mm
            mol = dico_atome[s.index]
            if(mol in dico_maxNegP):
                if(mp > dico_maxNegP[mol]):
                    dico_maxNegP[mol] = mp
            else:
                dico_maxNegP[mol] = mp
            if(mol in dico_maxNegM):
                if(mm > dico_maxNegM[mol]):
                    dico_maxNegM[mol] = mm
            else:
                dico_maxNegM[mol] = mm

        
        x = 0
        y = 0
        for key in dico_maxP:
            x = x + dico_maxP[key]
        for key in dico_maxM:
            y = y + dico_maxM[key]
        del dico_maxP
        del dico_maxM
        a = 0
        b = 0
        for key in dico_maxPsi:
            a = a + dico_maxPsi[key]
        for key in dico_maxMsi:
            b = b + dico_maxMsi[key]
        del dico_maxPsi
        del dico_maxMsi
        c = 0
        d = 0
        for key in dico_maxNegP:
            c = c + dico_maxNegP[key]
        for key in dico_maxNegM:
            d = d + dico_maxNegM[key]
        del dico_maxNegP
        del dico_maxNegM
        return x, y, a, b, c, d
    
    def nb_mol_ub(self,P):
        dico = {}
        li = P.S + P.SYes
        for s in li:
            if(s.classe == 1):
                dico[s.molecule] = 1
            else:
                dico[s.molecule] = 0

        P.nbClasse0 = 0
        P.nbClasse1 = 0
        for k in dico:
            if(dico[k] == 1):
                P.nbClasse1 = P.nbClasse1 + 1
            else:
                P.nbClasse0 = P.nbClasse0 + 1
        P.nbMol = len(dico)
        del dico
    
    def maj_si(self, P, ajout, item, v):
        global model       
       
        if(ajout):
            if(item):
                P.SIP, P.SIM = ds.calcul_si(P)
        else:
            if(item == False):
                P.SIP, P.SIM, P.SIPUB, P.SIMUB, P.SINegP, P.SINegM = ds.calcul_sietub(P)
            if(item):
                P.SIP, P.SIM, P.SIPUB, P.SIMUB, P.SINegP, P.SINegM = ds.calcul_sietub(P)
       

    def phi(self, P):
        global cible
        j = 0
        while(j < len(P.S)):
            u = P.S[j].index
            t = True
            deplace = False
            for k in P.P:
                i = k.index
                if(self.data[u][i] == False):
                    t = False
                    break
            if(t == True):
                pp = True
                for l in P.Pot:
                     i = l.index
                     if(self.data[u][i] == False):
                         pp = False
                         break
                if(pp):
                    ds.update_dlub(P,True,False,P.S[j])
                    P.SYes.append(P.S[j])
                    P.S.remove(P.S[j])

                    deplace = True
                if(deplace == False):
                    j = j + 1
            if(t == False):
                if(P.S[j].classe == 1):
                    P.nbClasse1 = P.nbClasse1 - 1
                               
                ds.update_dlub(P,False,False,P.S[j])
                P.SNot.append(P.S[j])
                P.S.remove(P.S[j])

        P.supp = len(P.S)+ len(P.SYes)
    
    def update_dlub(self,P,ajout,item,v):
        global weight1
        global weight0
        global alpha
        global beta
        if(item):
            if(ajout):
                P.DLP = P.DLP + alpha 
                P.DLM = P.DLM + alpha
            else:
                P.DLPUB = P.DLPUB - alpha
                P.DLMUB = P.DLMUB - alpha

    def psi(self, P):
        for k in P.Not:
            i = k.index
            t = True
            li = P.S + P.SYes
            for e in li:
                u = e.index
                if(self.data[u][i] == False):
                    t = False
                    break
            if(((len(P.S)+len(P.SYes)) > 0) and (t == True)):
                return False

        j = 0
        while(j < len(P.Pot)):
            i = P.Pot[j].index
            t = True
            li = P.S + P.SYes
            for e in li:
                u = e.index
                if(self.data[u][i] == False):
                    t = False
                    break
            if(t == True):
                P.P.append(P.Pot[j])
                ds.update_dlub(P,True,True,P.Pot[j])
                P.Pot.remove(P.Pot[j])
            else:
                 j = j + 1
        
        return True


def copy_P(dataset,P):
    Pat = list(P.P)
    Not = list(P.Not)
    Pot = list(P.Pot)
    S = list(P.S)
    SNot = list(P.SNot)
   
    Q = Pattern(Pat, S, Not, Pot,SNot)
    Q.SYes= list(P.SYes)
    Q.supp = P.supp
    Q.SIP = P.SIP
    Q.SIPUB = P.SIPUB
    Q.SIM = P.SIM
    Q.SIMUB = P.SIMUB
    Q.DLP = P.DLP
    Q.DLPUB = P.DLPUB
    Q.DLM = P.DLM
    Q.DLMUB = P.DLMUB
    Q.nbClasse1 = P.nbClasse1
    Q.nbClasse0 = P.nbClasse0
    Q.IR = P.IR
    Q.signe = P.signe
    Q.nbMol = P.nbMol
    Q.SINegM = P.SINegM
    Q.SINegP = P.SINegP
    return Q

def recopy_P(dataset, Q, P):
    del P.S
    P.S = list(Q.S)
    del P.P
    P.P = list(Q.P)
    del P.Pot
    P.Pot = list(Q.Pot)
    del P.SNot
    P.SNot = list(Q.SNot)
    del P.Not
    P.Not = list(Q.Not)
    del P.SYes
    P.SYes= list(Q.SYes)
    P.supp = Q.supp
    P.nbClasse1 = Q.nbClasse1
    P.nbClasse0 = Q.nbClasse0
    P.SIM = Q.SIM
    P.SIMUB = Q.SIMUB
    P.DLM = Q.DLM
    P.DLMUB = Q.DLMUB

    P.SIP = Q.SIP
    P.SIPUB = Q.SIPUB
    P.DLP = Q.DLP
    P.DLPUB = Q.DLPUB
    P.IR = Q.IR
    P.signe = Q.signe
    P.nbMol = Q.nbMol
    P.SINegM = Q.SINegM
    P.SINegP = Q.SINegP
    
def init_P(dataset):
    global Tclasse
    global Tmolecule
    global Tatome
    global Tnames
    global minsupp
    global idLayer
    global model
    global ds
    global weight0
    global weight1
    global dico_atome
    global cible
    global alpha
    global beta
    
    dico = {}
    P = Pattern([],[],[],[],[])
    for i in range(dataset.n):
        ind = Atome(i, Tclasse[i], Tatome[i], Tmolecule[i])
        if(ind.classe == 1):
            P.nbClasse1 =  P.nbClasse1 + 1
            dico[Tmolecule[i]] = 1
        else:
            dico[Tmolecule[i]] = 0
        P.S.append(ind)
        dico_atome[i] = Tmolecule[i]
        
    s = 0
    for k in dico:
        s = s + dico[k]
    dataset.pClasse1 = s
    P.nbClasse1 = s
    print("Classe+ : ",P.nbClasse1, "Classe- : ",len(dico) - P.nbClasse1, " Total :", len(dico))
    P.nbMol = len(dico)
    r = 0
    s = 0
    
    #DL
    for i in range(dataset.p):
        name = Tnames[i]
        if(idLayer != -1):
            d = name.find('_') + 1
            f = name.find('c')
            layer = int(name[d:f])
            d = f + 2
            component = int(name[d:len(name)])
            it = Item(i,layer,component,name)
        else:
            it = Item(i,0,0,name)
        P.Pot.append(it)

    P.DLP = P.DLP + beta
    P.DLM = P.DLM + beta
    P.DLPUB = alpha * len(P.Pot) + beta
    P.DLMUB = alpha * len(P.Pot) + beta
    dico_maxP = {}
    dico_maxM = {}
    for i in P.S:
        mp = 0
        mm = 0
        for j in P.Pot:
            if(i.classe == 1):
                mp = mp - weight1 * alog(model.proba[i.index][j.index])
            else:
                mm = mm - weight0 * alog(model.proba[i.index][j.index])
                
        mol = dico_atome[i.index]
        if(mol in dico_maxP):
            if(mp > dico_maxP[mol]):
                dico_maxP[mol] = mp
        else:
            dico_maxP[mol] = mp
        if(mol in dico_maxM):
            if(mm > dico_maxM[mol]):
                dico_maxM[mol] = mm
        else:
            dico_maxM[mol] = mm

    P.SIPUB = 0
    P.SIMUB = 0
    for key in dico_maxP:
        P.SIPUB = P.SIPUB + dico_maxP[key]
    for key in dico_maxM:
        P.SIMUB = P.SIMUB + dico_maxM[key]
    del dico_maxP
    del dico_maxM
    P.supp = len(P.S)+ len(P.SYes)
    compute_ir(P)
    return P

def SI_pattern(ds, P):
    global model
    global weight1
    global weight0
    dico_maxP = {}
    dico_maxM = {}
    li = P.S + P.SYes
    for j in li:
        mp = 0
        mm = 0
        for i in P.P:
            if(j.classe == 1):
                mp = mp - weight1 * alog(model.proba[j.index][i.index])
            else:
                mm = mm - weight0 * alog(model.proba[j.index][i.index])
        mol = dico_atome[j.index]
        if(mol in dico_maxP):
            if(mp > dico_maxP[mol]):
                dico_maxP[mol] = mp
        else:
            dico_maxP[mol] = mp
        if(mol in dico_maxM):
            if(mm > dico_maxM[mol]):
                dico_maxM[mol] = mm
        else:
            dico_maxM[mol] = mm
       
    r = 0
    s = 0
    for key in dico_maxP:
        r = r + dico_maxP[key]
    for key in dico_maxM:
        s = s + dico_maxM[key]
    del dico_maxP
    del dico_maxM
    return r,s

def compute_ub(P):
    global minIR
    global cible
    global inf
    if(cible == 1):
        if(P.DLP > 0):
            if(P.DLMUB > 0):
                r = (1.0 /P.DLP) *(float(P.SIPUB) - float(P.SINegM))
            else:
                r = (float(P.SIPUB))/P.DLP 
        else:
            r = inf
    else:
        if(P.DLM > 0):
            if(P.DLPUB > 0):
                r = (1.0 /P.DLM) *(float(P.SIMUB) - float(P.SINegP))
            else:
                r = (P.SIMUB/float(P.DLM)) 
        else:
            r = inf
    return r 

def compute_ir(P):
    global cible
    global inf
    if(cible == 1):
         if(P.DLP > 0):
            if(P.DLM > 0):
                P.IR = (float(P.SIP)/P.DLP) - (float(P.SIM)/P.DLM)
                P.signe = 1 
            else:
                P.IR = -inf
         else:
             P.IR = 0
    else:
        if(P.DLM > 0):
            if(P.DLP > 0):
                P.IR = (float(P.SIM)/P.DLM) - (float(P.SIP)/P.DLP)
                P.signe = 0 
            else:
                P.IR = -inf
        else:
            P.IR = 0


    
def SGEnum(ds, depth):
    global pile_pattern
    global SGset
    global minsupp
    global topk
    global nbenum
    global prune
    global minIR
    global cible

    nbenum = nbenum + 1
    P = pile_pattern[depth]
    if(nbenum % 1000 == 0):
        print(nbenum, " minir ", minIR, "---", P.SIP," ",P.DLP," ",P.IR," ",len(P.Pot))
    ds.nb_mol_ub(P)
    if(cible == 1):
        molSupp = P.nbClasse1
    else:
        molSupp = P.nbMol - P.nbClasse1
    
    ds.phi(P)
    ds.maj_si(P, False, False, 0)
    ub = compute_ub(P)
    
    if ((ub < minIR)):
        prune = prune + 1
    if(molSupp < minsupp):
        return
    if ((P.supp < minsupp) or (ub < minIR) or (ds.psi(P) == False)):
        return
    
    if(len(P.Pot) == 0):
        for v in P.S:
            ds.update_dlub(P, True, False, v)
        
        compute_ir(P)
        if((topk != None) and (len(SGset) >= topk)):
            SGset = sorted(SGset, key = lambda x: (x.IR))
            minIR = SGset[0].IR
            if(P.IR >= minIR):
                C = copy_P(ds,P)
                SGset[0] = C
        else:
            if (P.IR >= minIR):
                C = copy_P(ds,P)
                SGset.append(C)
        return
    else:    
        if(len(pile_pattern) > depth + 1):
            NewP = pile_pattern[depth + 1]
            recopy_P(ds, P, NewP)        
        else:
            NewP = copy_P(ds, P)
            pile_pattern.append(NewP)
        
        index = NewP.Pot.pop()
        NewP.P.append(index)
      
        ds.update_dlub(NewP, True, True, index)

        SGEnum(ds, depth + 1)

        recopy_P(ds, P, pile_pattern[depth+1])
        pile_pattern[depth+1].Not.append(index)
        pile_pattern[depth+1].Pot.remove(index)
        ds.update_dlub(pile_pattern[depth+1], False, True, index)
        compute_ir(pile_pattern[depth+1])
        
        SGEnum(ds, depth + 1)
        

              
def print_P(ds,P):
    ds.nb_mol_ub(P)
    print("(",P.nbMol, " x ", len(P.P), end=') '),
    print("", end='('),
    print(P.nbClasse1," ",P.nbMol - P.nbClasse1, end=') '),
    sip, sim = SI_pattern(ds, P)
    print(" si ",truncate(sip)," ",truncate(P.SIP), " ", truncate(P.SIPUB), " || ", sim," ",truncate(P.SIM)," ",truncate(P.SIMUB)," DL ",truncate(P.DLP)," ", truncate(P.DLPUB)," ", truncate(P.DLM)," ", truncate(P.DLMUB), " ratio", truncate(P.IR), P.signe)

    
def lecture_fichier(nom):
    print(nom)
    print("File sorted by graph ids")
    df = pd.read_csv(nom, index_col = 0)
    c = df.columns
    cols = c[1:len(c)-2]
    a = df[cols]
    ord = {}
    for i in cols:
        ord[i] = np.sum(a[i].values > 0)
    ord = dict(sorted(ord.items(), key=lambda item: item[1], reverse=True))
    c_ord = ord.keys()
    a = a[c_ord]
    
    v_classe = c[len(c)-1]
    classe = df[v_classe]
    
    nb_mol = len(df.groupby(by=["id"]).count())
    arome = df.groupby(by=["id","class"]).count()
    nb_mol_plus = len(arome.loc[(slice(None), 1), :])
    v_molecule = c[0]
    molecule = df[v_molecule]
    v_atome = c[len(c)-2]
    atomes = df[v_atome]
    ## compute model
    data = a.values > 0
    pligne = np.sum(data,axis = 1) / data.shape[1]
    pcol = np.sum(data,axis = 0) / data.shape[0]
    pcol = np.array(pcol)
    pligne = np.array(pligne)
    return data, classe, molecule, atomes, a.columns, pcol, pligne, nb_mol, nb_mol_plus

def b(ds):
    global Tmolecule
   
    indices = []
    ind = 0
    i = 0
    for i in range(ds.p):
        s = np.sum(ds.data[:,i])
        garde = False
        if(s >= minsupp):
            name = Tnames[i]
            if(idLayer != -1):
                d = name.find('_') + 1
                f = name.find('c')
                layer = int(name[d:f])
            if((idLayer == -1) or (layer == idLayer)):
                garde = True
                indices.append(i)
    ds.p = len(indices)
    val = np.zeros((ds.n,ds.p))
    for i in range(ds.n):
        for j in range(len(indices)):
            val[i][j] = ds.data[i][indices[j]]
    pc = np.zeros(ds.p)
    for j in range(len(indices)):
        pc[j] = ds.pcol[indices[j]]
    del(ds.data)
    del(ds.pcol)
    ds.pcol = pc
    ds.data = val
   
              
def print_SG(ds):
    global SGset
    if(len(SGset) > 0):
        for i in range(len(SGset)):
            print_P(ds,SGset[i])
        
def print_SG_file(ds,resultat):
    global f_res
    global cible
    dico = {}
    if(len(resultat) > 0):
        for k in range(len(resultat)):
            P = resultat[k]
            ds.nb_mol_ub(P)

            
            f_res.write("# "+str(k)+" target:"+ str(cible) + " c+:"+str(P.nbClasse1)+" c-:"+str(P.nbMol - P.nbClasse1)+" score:"+str(P.IR)+" score+:"+str(P.SIP)+" score-:"+str(P.SIM)+" nb:"+str(len(P.P))+" =")
            for i in P.P:
                f_res.write(str(i.name)+' '),
            f_res.write("\n"),
            li = P.S + P.SYes
            for i in li:
                if(i.classe == 1):
                    dico[i.molecule] = 1
                else:
                    dico[i.molecule] = 0
                f_res.write(str(int(i.atome))+' '+str(int(i.molecule))+' '+str(i.classe)+"\n"),
            f_res.write(" fin motif\n"),
            
    neg = 0
    pos = 0
    for k in dico:
        if(dico[k] == 0):
            neg = neg + 1
        else:
            pos = pos + 1
    print("couverture ",len(dico)," pos ", pos," neg ",neg)

def init_piles(dataset, P):
    global pile_pattern
    global supp
    global topk
    global nbenum
    global prune
    del pile_pattern
    pile_pattern = []
    pile_pattern.append(P)
    
       
if __name__ == "__main__":

        
    parser = OptionParser()
    parser.add_option("-i", "--donnees", dest="donnees",
                      help="chemin du fichier d'instances", metavar="FILE")
    parser.add_option("-s", "--support", dest="support",
                      help="le support", metavar="int")
    parser.add_option("-l", "--layer", dest="layer",
                      help="le layer", metavar="int")
    parser.add_option("-m", "--cible", dest="cible",
                      help="la cible positive ou negative", metavar="int")
    parser.add_option("-k", "--topk", dest="tpk",
                      help="TopK", metavar="int")
    (options, args) = parser.parse_args()

    if ((not options.donnees) or (not options.support)):  # if filename is not given
        parser.error('-i (--donnees) option required')
        parser.error('-s (--support) option required')
    else:
        nom_donnees = options.donnees
        minIR = 0.0
        if(options.cible):
            cible = int(options.cible)
        else:
            cible = 1
        minsupp = int(options.support)
        if(options.layer):
            idLayer = int(options.layer)
        else:
            idLayer = -1
        if(options.tpk):
            topNbK = int(options.tpk)
        
        #### fichiers
        nd = nom_donnees.find(".")
        nd = nom_donnees[0:nd]
        f_res = open(nd+"_lay_"+str(idLayer)+"_cib_"+str(cible)+"res.txt", "w")
        f_perf = open(nd+"_"+str(idLayer)+"_perf.txt", "w")
        
        print("cible ",cible)
        data, Tclasse, Tmolecule, Tatome, Tnames, pcol, pligne, nb_mol, nb_mol_plus = lecture_fichier(nom_donnees)
        #print(nb_mol, nb_mol_plus)
        n_samples0 = nb_mol - nb_mol_plus
        n_samples1 = nb_mol_plus
        weight1 = 1.0
        weight0 = 1.0
        
        if(n_samples0 > n_samples1):
            weight0 = 1.0
            weight1 = float(n_samples0)/n_samples1
        else:
            weight0 = float(n_samples1)/n_samples0
            weight1 = 1.0
        #print("les poids", weight0, weight1)
        ds = Dataset(data, pcol, pligne)
        ds.prob = np.sum(ds.data)/float(ds.n * ds.p)
        b(ds)
        model = Model(pcol,pligne)
        model.compute_model()
        
        topk = 1
        resultat = []
        
        i = 0
        si = 10000
        while (si > 10.0) and ( i < 10):
            print("iter ", i)
            P = init_P(ds)
            init_piles(ds, P)
            SGEnum(ds, 0)
            print_SG(ds)
            si = 0
            if(len(SGset) > 0):
                C = copy_P(ds,SGset[0])
                if(C.IR > 0):
                    resultat.append(C)
                    del SGset[0]
                    si = C.IR
            
            print("nbEnum", nbenum, "nbprune", prune, "nbmotifs", len(resultat))
            SGset = []
            minIR = 0.0
            model.maj(C)
            nbenum = 0
            prune = 0
            i = i + 1
            
        print_SG_file(ds,resultat)
        f_res.close()
        f_perf.close()
