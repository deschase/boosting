# -*- coding: utf-8 -*-
from arbre import *
from readFiles import *

class DiscreteAdaboost:
    def __init__(self, nbData, nbClassifieur):
        self.nbdata = nbData
        self.nbclassifieur = nbClassifieur
        self.trees = []
        self.coefs = []
        self.errors = []

    def fit(self, data, y):
        longeur = len(data)
        self.w = [1./longeur for i in range(0,longeur)]
        for m in range(0,self.nbclassifieur):
            # On crée et on entraine l'arbre
            arbre = create_tree(data,y,self.w)
            self.trees.append(arbre)
            #On calcule l'erreur
            pred = arbre.predict(data)
            vectdiff = []
            for k in range(0,longeur):
                if pred[k] == y[k]:
                    vectdiff.append(0)
                else:
                    vectdiff.append(1)
            erreur = np.dot(self.w,vectdiff)
            self.errors.append(erreur)
            # On calcule les coefficients d'importance de l'arbre
            if erreur == 0:
                c = 1
            elif erreur == 1:
                c = 0
            else:
                c = math.log((1-erreur)/erreur)
            self.coefs.append(c)
            # On modifie les poids
            somme = 0
            for i in range(0,longeur):
                self.w[i] = self.w[i]*math.exp(c*vectdiff[i])
                somme += self.w[i]
            for i in range(0,self.nbdata):
                self.w[i] = self.w[i]/somme
            #print "score arbre {}".format(m),arbre.score(data,y)

    def predict(self, data):
        summ = np.zeros(len(data))
        for i in range(self.nbclassifieur):
            p = self.trees[i].predict(data)
            for j in range(len(data)):
                summ[j]= summ[j] + self.coefs[i]*p[j]
        return (summ >= 0)*1 + (summ < 0)*(-1)

    def score(self, data, label):
        pred = self.predict(data)
        return np.mean((pred == label))


data, y = donneData("database/wdbc.data",2,1,True,0,True)
ada = DiscreteAdaboost(len(data)/2, 400)
data_moit = data[0:ada.nbdata,:]
y_moit = y[0:ada.nbdata]
ada.fit(data_moit,y_moit)
print "score final = ", ada.score(data, y)
