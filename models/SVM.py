import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

#start with Xtrain,Xtest=read('train') ,read('test')
#then Xtrain,Xtest=scale(Xtrain,Xtest)
#then pred = SVM(Xtrain,Xtest)

filename="/home/raphael/Documents/dreem-challange-main/data/interim/"



def auto():
    Xtrain,Xtest=read('train') ,read('test')
    Xtrain,Xtest=scale(Xtrain,Xtest)
    pred = SVM(Xtrain,Xtest)
    return pred

def read(mode='train',RC=False):
    pulse=np.load(filename+'lpulse'+str(mode)+'.npy',allow_pickle=True)
    entropy=np.load(filename+'entropy'+str(mode)+'.npy',allow_pickle=True)
    freq=np.load(filename+'fbands'+str(mode)+'.npy',allow_pickle=True)
    entreegs=np.load(filename+'entreeg'+str(mode)+'.npy',allow_pickle=True)
    minmaxvarray=np.load(filename+'minmaxvarray'+str(mode)+'.npy',allow_pickle=True)
    MMD=np.load(filename+'MMD'+str(mode)+'.npy',allow_pickle=True)
    fmax=np.load(filename+'fmax'+str(mode)+'.npy',allow_pickle=True)
    fmax=np.reshape(fmax,(fmax.shape[0],fmax[0].size))
    PFD=np.load(filename+'PFD'+str(mode)+'.npy',allow_pickle=True)
    if RC==True:
        if mode=='train':
            RRtrain=np.load(filename+'bestfeatRtrain.npy',allow_pickle=True)
            print(pulse.shape,fmax.shape,entropy.shape,RRtrain.shape)
            return np.concatenate((pulse,entropy,freq,entreegs,minmaxvarray,MMD,fmax,PFD,RRtrain),axis=1)
        else:
            RRtest=np.load(filename+'bestfeatRest.npy',allow_pickle=True)
            print(pulse.shape,fmax.shape,entropy.shape,RRtest.shape)
            return np.concatenate((pulse,entropy,freq,entreegs,minmaxvarray,MMD,PFD,fmax,RRtest),axis=1)
    return np.concatenate((pulse,entropy,freq,entreegs,minmaxvarray,MMD,fmax,PFD),axis=1)


def scale(Xtrain,Xtest):
    scaler = StandardScaler()
    scaler.fit(Xtrain)
    Xtrain=scaler.transform(Xtrain)
    Xtest=scaler.transform(Xtest)
    return Xtrain,Xtest


def SVM(Xtrain,Xtest,C=20,save=True):
    SVM=SVC(C=C)
    y=readlabel()
    SVM.fit(Xtrain,y)
    pred=SVM.predict(Xtest)
    if save==True:
         pd.DataFrame(pred).to_csv("/media/raphael/Data/Dataaccess/ML_DREEM/predtest.csv")
    return pred


def readlabel():
    filename = "/media/raphael/Data/Dataaccess/ML_DREEM/y_train.csv"
    ytrain=np.array(pd.read_csv(filename))
    for i in range(5):
        print('occurence sleep stage',i,np.count_nonzero(ytrain[:,1]== i))
    y=ytrain[:,1]
    return y


