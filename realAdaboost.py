from arbre import *
from readFiles import *

class RealAdaboost:
    def __init__(self, nbData, nbClassifieur):
        self.nbdata = nbData
        self.nbclassifieur = nbClassifieur
        W = np.zeros(nbData)
        for i in range(nbData):
            W[i] = float(1)/float(nbData)
        self.w = W 
        #print W
        self.trees = []

    def fit(self, data, label):
        for j in range(self.nbclassifieur):
            tree = create_tree(data, label, self.w)
            (self.trees).append(tree)
            p = tree.predict_proba(data)
            print j, ": ", tree.score(data,y)
            summ = 0
            for i in range(self.nbdata):
            	if(p[i][0] > 0 and p[i][0] < 1):
                    self.w[i] = self.w[i]*math.exp(-label[i]* (1/2)* math.log(p[i][0]/(1-p[i][0])))
                   # self.w[i] = 0.5
                   # print self.w[i]
                else:
                	#print "out"
                	self.w[i]  = float(1)/float(self.nbdata)


                summ += self.w[i]
            #print summ

            if(summ != 0):
                self.w = (1/summ)*(self.w)

            #print self.w
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
ada = RealAdaboost(len(data), 100)
ada.fit(data,y)
print ada.score(data, y)
        

