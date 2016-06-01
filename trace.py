from realAdaboost import *
from discreteAdaboost import *
from readFiles import *
from logitBoost import *

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
            ada = DiscreteAdaboost(len(data)/2, i)
            data_moit = data[0:ada.nbdata,:]
            y_moit = y[0:ada.nbdata]
            ada.fit(data_moit,y_moit)
            tab.append(ada.score(data,y))
        if boostChoisi == "LogitBoost":
            ada = LogitBoost(len(data)/2, i)
            data_moit = data[0:ada.nbdata,:]
            y_moit = y[0:ada.nbdata]
            ada.fit(data_moit,y_moit)
            tab.append(ada.score(data, y,ada.nbclassifieur))

    plt.figure()
    plt.plot(range(depart,arrivee,pas), tab)
    plt.xlabel('nombre de classifieur')
    plt.ylabel('score')
    if boostChoisi == "RealAdaboost":
        plt.title('Evolution du score du Real Adaboost (donnees cancer)')
    if boostChoisi == "DiscreteAdaboost":
        plt.title('Evolution du score du Discrete Adaboost (donnees cancer)')
    if boostChoisi == "LogitBoost":
        plt.title('Evolution du score du LogitBoost (donnees cancer)')
    plt.grid(True)
    if boostChoisi == "RealAdaboost":
        plt.savefig("realada.png")
    if boostChoisi == "DiscreteAdaboost":
        plt.savefig("discreteada.png")
    if boostChoisi == "LogitBoost":
        plt.savefig("logitBoost.png")
    plt.show()

data, y = donneData("database/wdbc.data", 2,1, True, 0 )

trace("LogitBoost", 1, 300, 10, data, y)
