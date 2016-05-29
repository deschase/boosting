
import numpy as np

def ouvreEtLit(nomfichier):
    # Renvoie un tableau avec toutes les donnees par lignes
    tableau = []
    with open(nomfichier, 'r') as f:
        data = f.readlines()

        for line in data:
            words = line.rstrip('\n').split(',')
            if words != ['']:
                tableau.append(words)

    return np.asarray(tableau)

def donneData(nomfichier, nbLabel = 2, colonne = 4, suppress = False, colonneSup = 0):
    tableau = ouvreEtLit(nomfichier)
    labelText = tableau[:, colonne]
    data = np.delete(tableau, colonne, axis=1)
    if suppress:
        data = np.delete(data, colonneSup, axis=1)
    data2 = []
    for i in range(0,len(data)):
        data2.append(map(float, data[i]))
    data2 = np.asarray(data2)
    labelInt = {}
    i = -1
    for label in labelText:
        if label not in labelInt:
            labelInt[label] = i
            if(i+1 < nbLabel and i != -1):
                i+=1
            if(i == - 1):
                i += 2

    labelTrue = []

    for label in labelText:
        labelTrue.append(labelInt[label])

    return data, labelTrue



