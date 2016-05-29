
import numpy as np

def ouvreEtLit(nomfichier):
	# Renvoie un tableau avec toutes les donnees par lignes 
	tableau = []
    with open(nomfichier, 'r') as f:
        data = f.readlines()

        for line in data:
            words = line.split(',')
            tableau.append(words)

    return np.asarray(tableau)

def donneData(nomfichier, nbLabel = 2, colonne = 4, suppress = False, colonneSup = 0):
	tableau = ouvreEtLit(nomfichier)
	labelText = tableau[:, colonne]
	data = numpy.delete(tableau, colonne, axis=1)
	if suppress:
		data = numpy.delete(data, colonneSup, axis=1)
	labelInt = {}
	i = 0
	for label in labelText:
        if label not in labelInt:
        	labelInt[label] = i
        	if(i+1 < nbLabel):
        	    i+=1
    labelTrue = []

    for label in labelText:
    	labelTrue.append(labelInt[label])

    return data, labelTrue



