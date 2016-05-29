from arbre import *
from readFiles import *
import math

class RealAdaboost:
    def __init__(self, nbData, nbClassifieur):
        self.nbdata = nbData
        self.nbclassifieur = nbClassifieur
        W = np.zeros(data.shape[0])
        for i in range(data.shape[0]):
            W[i] = 1/data.shape[0]
        self.w = W 
        self.trees = []

    def fit(self, data, label):
        for j in range(self.nbclassifieur):
            tree = create_tree(data, label, self.w)
            (self.trees).append(tree)
            p = tree.predict_proba(data)
            summ = 0
            for i in range(self.nbdata):
            	if(p[i][0] > 0):
                    self.w[i] = self.w[i]*math.exp(-label[i]* (1/2)* math.log(p[i][1]/(1-p[i][1])))
                else:
                	
                summ += self.w[i]

            self.w = self.w/summ
        return 0 

    def predict(self, data):
        summ = self.trees[0].predict(data)
        for i in range(1,self.nbclassifieur):
            summ = summ + self.trees[i].predict(data)
        return (summ >= 0)*1 + (summ < 0)*(-1)

    def score(self, data, label):
        pred = self.predict(data)
        return np.mean((pred == label))

            
        
data, y = donneData("database/iris.data")
print y
ada = RealAdaboost(len(data), 10)
ada.fit(data,y)
print score(data, y)
        

