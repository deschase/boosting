# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt # module pour les outils graphiques
import tools # module fourni en TP1
from sklearn import tree # module pour les arbres
from sklearn import ensemble # module pour les forets
from sklearn import cross_validation as cv
from IPython.display import Image
import pydot


#Initialisation
data,y=tools.gen_arti(1)
N=len(y)
w_init=[1./N for i in range(0,N)]

def create_tree(data, y, vect_poid=w_init, prof=1, split=1):
    mytree=tree.DecisionTreeClassifier() #creation d'un arbre de decision
    mytree.max_depth=1 #profondeur maximale de 5
    mytree.min_samples_split=1 #nombre minimal d'exemples dans une feuille
    #Apprentissage
    mytree.fit(data,y,sample_weight=vect_poid)
    return mytree

mytree = create_tree(data,y)
#prediction
pred=mytree.predict(data)
print "precision : ", 1.*(pred!=y).sum()/len(y)

#ou directement pour la precision :
print "precision (score) : "  +` mytree.score(data,y)`

#Importance des variables :
plt.subplot(1,2,2)
plt.bar([1,2],mytree.feature_importances_)
plt.title("Importance Variable")
plt.xticks([1,2],["x1","x2"])

#Affichage de l'arbre
with file("mytree.dot","wb") as f:
    tree.export_graphviz(mytree,f)

###### Si graphviz n'est pas installe, la fonction suivante permet d'afficher un arbre
def affiche_arbre(tree):
    long = 10
    sep1="|"+"-"*(long-1)
    sepl="|"+" "*(long-1)
    sepr=" "*long
    def aux(node,sep):
        if tree.tree_.children_left[node]<0:
            ls ="(%s)" % (", ".join( "%s: %d" %(tree.classes_[i],int(x)) for i,x
 in enumerate(tree.tree_.value[node].flat)))
            return sep+sep1+"%s\n" % (ls,)
        return (sep+sep1+"X%d<=%0.2f\n"+"%s"+sep+sep1+"X%d>%0.2f\n"+"%s" )% \
                    (tree.tree_.feature[node],tree.tree_.threshold[node],aux(tree.tree_.children_left[node],sep+sepl),
                    tree.tree_.feature[node],tree.tree_.threshold[node],aux(tree.tree_.children_right[node],sep+sepr))
    return aux(0,"")
print(affiche_arbre(mytree))
# plt.figure()
# tools.plot_frontiere(data, mytree.predict, 50)
# tools.plot_data(data,y)
# plt.show()
