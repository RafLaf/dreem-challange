import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, chi2, f_classif, f_regression
from sklearn.decomposition import PCA

filename="../../data/interim/"


def read(mode='train'):
    R=np.load(filename+'R'+str(mode)+'.npy',allow_pickle=True)
    return R

def readlabel():
    filename = "../../data/raw/y_train.csv"
    ytrain=np.array(pd.read_csv(filename))
    for i in range(5):
        print('occurence sleep stage',i,np.count_nonzero(ytrain[:,1]== i))
    y=ytrain[:,1]
    return y

def bestfeat(save=True, method=f_classif,k=50):
    #method = chi2 or f_classif or f_regression others available on the sckit learn website
    
    Rtrain=read(mode='train')
    Rtest=read(mode='test')
    y=readlabel()

    if method==PCA:
        clf = PCA(n_components=k) 
    else:
        clf=SelectKBest(method, k=k)

    clf.fit(Rtrain, y)
    Xtrain=clf.transform(Rtrain)
    Xtest=clf.transform(Rtest)
    if save==True:
        np.save(filename+'bestfeatRtrain.npy',Xtrain)
        np.save(filename+'bestfeatRtest.npy',Xtest)
    # clf.scores_.sort()
    # return Xtrain,Xtest,clf.scores_

if __name__ == "__main__":
    bestfeat(save=True, method=PCA, k=5)


