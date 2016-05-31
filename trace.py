from realAdaboost import *
from discreteAdaboost import *
from readFiles import *

def trace(boostChoisi, depart, arrivee, pas, data, y):
    #le boostchoisi doit etre un str
    tab = []
    for i in range(depart, arrivee, pas):
        print i , "sur ", arrivee
        if boostChoisi == "RealAdaboost":
            ada = RealAdaboost(len(data),i)
            ada.fit(data,y)
            tab.append(ada.score(data,y))
        if boostChoisi == "DiscreteAdaboost":
            ada = DiscreteAdaboost(len(data),i)
            ada.fit(data,y)
            tab.append(ada.score(data,y))

    plt.figure()
    plt.plot(range(depart,arrivee,pas), tab)
    plt.xlabel('nombre de classifieur')
    plt.ylabel('score')
    plt.title('Evolution du score du Real Adaboost (donnees cancer)')
    plt.grid(True)
    plt.savefig("realada2.png")
    plt.show()

data, y = donneData("database/wdbc.data", 2,1, True, 0 )

trace("RealAdaboost", 1, 1500, 10, data, y)
