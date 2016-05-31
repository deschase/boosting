from arbre import *
from readFiles import *
import matplotlib.pyplot as plt

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
                    self.w[i] = self.w[i]*math.exp(-label[i]*(float(1)/float(2))*math.log(p[i][1]/(1-p[i][1])))
                    
                else:
                    
                    self.w[i]  = 0#float(1)/float(self.nbdata)


                summ += self.w[i]
            #print summ

            if(summ != 0):
                self.w = (1/summ)*(self.w)

            #print self.w
        return 0 

    def predict(self, data):
        summ = np.zeros(data.shape[0])
        for i in range(self.nbclassifieur):
            p = self.trees[i].predict_proba(data)
            for j in range(self.nbdata):
                if(p[j][0] > 0 and p[j][0] < 1):
                    summ[j]= summ[j] + (float(1)/float(2))*math.log(p[j][1]/(1-p[j][1]))
                else :
                    summ[j] = summ[j] + (1/2)*p[j][1] + (-1/2)*p[j][0]

        return (summ >= 0)*1 + (summ < 0)*(-1)

    def score(self, data, label):
        pred = self.predict(data)
        return np.mean((pred == label))

            
# omfichier, nbLabel = 2, colonne = 4, suppress = False, colonneSup = 0):       
data2, y2 = donneData("database/wdbc.data", 2,1, True, 0 )
print y2
ada = RealAdaboost(len(data2)/2,600)
data_moit = data2[0:ada.nbdata,:]
y_moit = y2[0:ada.nbdata]
ada.fit(data_moit,y_moit)

print "score final = ", ada.score(data, y)




