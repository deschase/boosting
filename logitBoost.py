# -*- coding: utf-8 -*-
from readFiles import *

class LogitBoost:
    def __init__(self, nbData, nbClassifieur):
        self.nbdata = nbData
        self.nbclassifieur = nbClassifieur
        self.w = [1./nbData for i in range(0,nbData)]
        self.z = [0.0 for i in range(0,nbData)]
        self.p = [0.5 for i in range(0,nbData)]
        sefl.sol = [0.0 for i in range(0,nbData)]

    def fit(self, data, y):
        for m in range(0,self.nbclassifieur):
            # On calcule les z_i et les w_i
            for k in range(0,self.nbdata):
                y_star = int((y[k] + 1)/2)
                z[k] = y_star
                    vectdiff.append(0)
                else:
                    vectdiff.append(1)

            # On calcule l'erreur
            pred = arbre.predict(data)
            vectdiff = []
            for k in range(0,self.nbdata):
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
            for i in range(0,self.nbdata):
                self.w[i] = self.w[i]*math.exp(c*vectdiff[i])
                somme += self.w[i]
            for i in range(0,self.nbdata):
                self.w[i] = self.w[i]/somme
            print "score",arbre.score(data,y)

    def scoretot(self, data, label):
        results = []
        for x in range(0,self.nbdata):
            resx = 0
            for j in range(0,self.nbclassifieur):
                resx += self.coefs[j]*(self.trees[j].predict(data))[x]
            if resx < 0:
                results.append(-1)
            else:
                results.append(1)
        n = len(label)
        res = 0
        for i in range(0,n):
            if results[i] == label[i]:
                res += 1
        return res/float(n)

data, y = donneData("database/wdbc.data",2,1,True,0,True)
ada = DiscreteAdaboost(len(data), 10)
ada.fit(data,y)
print "score final = ", ada.scoretot(data, y)
