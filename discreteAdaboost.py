# -*- coding: utf-8 -*-
from arbre import *
from readFiles import *

#Données
nomfichier="database/iris.data"
data,y=donneData(nomfichier)

def discreteAdaboost():
    # Nombre d'échantillons
    N=len(y)
    # Vecteur des poids intial
    w=[1./N for i in range(0,N)]
    list_w=[]
    list_w.append(w)
    # Nombre de d'arbres
    M = 10
    # Liste des arbres et de coefs associés
    list_arbre = []
    list_coef = []
    # Evolution de l'erreur
    list_erreur = []
    for m in range(0,M):
        # On crée et on entraine l'arbre
        arbre = create_tree(data,y,w)
        list_arbre.append(arbre)
        #On calcule l'erreur
        pred = arbre.predict(data)
        vectdiff = []
        for k in range(0,N):
            if pred[i] == y[i]:
                vectdiff.append(0)
            else:
                vectdiff.append(1)
        erreur = np.dot(w,vectdiff)
        list_erreur.append(erreur)
        # On calcule les coefficients d'importance de l'arbre
        c = log((1-erreur)/erreur)
        list_coef.append(c)
        # On modifie les poids
        somme = 0
        for i in range(0,N):
            w[i] = w[i]*exp(c*vectdiff)
            somme += w[i]
        for i in range(0,N):
            w[i] = w[i]/somme
    results = []
    for x in range(0,N):
        resx = 0
        for j in range(0,M):
            resx += c[j]*(list_arbre[j].predict(data)[x])
        results.append(resx)
    return results

