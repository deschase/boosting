# -*- coding: utf-8 -*-
from readFiles import *
from arbre import *

class LogitBoost:
    def __init__(self, nbData, nbClassifieur):
        self.nbdata = nbData
        self.nbclassifieur = nbClassifieur
        self.trees = []
        self.w = [1./nbData for i in range(0,nbData)]
        self.z = [0.0 for i in range(0,nbData)]
        self.p = [0.5 for i in range(0,nbData)]
        self.sol = [0.0 for i in range(0,nbData)]

    def fit(self, data, y):
        for m in range(0,self.nbclassifieur):
            # On calcule les z_i et les w_i
            for k in range(0,self.nbdata):
                y_star = int((y[k] + 1)/2)
                if self.p[k] == 1.:
                    self.z[k] == 0.
                else:
                    self.z[k] = (y_star-self.p[k])/(self.p[k]*(1-self.p[k]))
                self.w[k] = self.p[k]*(1-self.p[k])
            # On fit le classifier
            arbre = create_tree_logit(data, self.z, self.w)
            self.trees.append(arbre)
            # On met à jour la fonction de classement globale et les probabilités
            pred = arbre.predict(data)
            for k in range(0,self.nbdata):
                self.sol[k] += 0.5*pred[k]
            for k in range(0,self.nbdata):
                self.p[k] = math.exp(self.sol[k])/(math.exp(self.sol[k]) + math.exp(-self.sol[k]))
            print "score arbre {}".format(m),self.score(data,y)

    def score(self, data, label):
        res = 0
        for i in range(0,self.nbdata):
            if self.sol[i] >= 0:
                if label[i] == 1:
                    res += 1
            else:
                if label[i] == -1:
                    res += 1
        return res/float(self.nbdata)

data, y = donneData("database/wdbc.data",2,1,True,0,True)
ada = LogitBoost(len(data), 30)
ada.fit(data,y)
print "score final = ", ada.score(data, y)
