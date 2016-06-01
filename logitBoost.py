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
            # print "score arbre {}".format(m),self.score(data,y,m)

    def predict(self, data, quant):
        summ = np.zeros(len(data))
        for i in range(quant):
            p = self.trees[i].predict(data)
            for j in range(len(data)):
                summ[j]= summ[j] + p[j]
        return (summ >= 0)*1 + (summ < 0)*(-1)

    def score(self, data, label, quant):
        pred = self.predict(data, quant)
        return np.mean((pred == label))


data, y = donneData("database/wdbc.data",2,1,True,0,True)
ada = LogitBoost(len(data)/2, 200)
data_moit = data[0:ada.nbdata,:]
y_moit = y[0:ada.nbdata]
ada.fit(data_moit,y_moit)
print "score final = ", ada.score(data, y,ada.nbclassifieur)

