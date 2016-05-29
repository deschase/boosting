
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