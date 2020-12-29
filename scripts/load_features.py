import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

PATH="../data/interim/"

def scale(Xtrain,Xtest=[]):
    scaler = StandardScaler()
    scaler.fit(Xtrain)
    Xtrain=scaler.transform(Xtrain)

    if len(Xtest) == 0:
        return Xtrain

    Xtest=scaler.transform(Xtest)
    return Xtrain,Xtest

def readlabel():
    filename = "../data/raw/y_train.csv"
    ytrain=np.array(pd.read_csv(filename))
    for i in range(5):
        print('occurence sleep stage',i,np.count_nonzero(ytrain[:,1]== i))
    y=ytrain[:,1]
    return y

def read(mode='train',RC=False):
    pulse=np.load(PATH+'lpulse'+str(mode)+'.npy',allow_pickle=True)
    entropy=np.load(PATH+'entropy'+str(mode)+'.npy',allow_pickle=True)
    freq=np.load(PATH+'fbands'+str(mode)+'.npy',allow_pickle=True)
    entreegs=np.load(PATH+'entreeg'+str(mode)+'.npy',allow_pickle=True)
    minmaxvarray=np.load(PATH+'minmaxvarray'+str(mode)+'.npy',allow_pickle=True)
    MMD=np.load(PATH+'MMD'+str(mode)+'.npy',allow_pickle=True)
    fmax=np.load(PATH+'fmax'+str(mode)+'.npy',allow_pickle=True)
    fmax=np.reshape(fmax,(fmax.shape[0],fmax[0].size))
    PFD=np.load(PATH+'PFD'+str(mode)+'.npy',allow_pickle=True)
    breath=np.load(PATH+'breath'+str(mode)+'.npy',allow_pickle=True)
    if RC==True:
        if mode=='train':
            RRtrain=np.load(PATH+'bestfeatRtrain.npy',allow_pickle=True)
            print(pulse.shape,fmax.shape,entropy.shape,RRtrain.shape)
            return np.concatenate((pulse,entropy,freq,entreegs,minmaxvarray,MMD,PFD,fmax,breath,RRtrain),axis=1)
        else:
            RRtest=np.load(PATH+'bestfeatRest.npy',allow_pickle=True)
            print(pulse.shape,fmax.shape,entropy.shape,RRtest.shape)
            return np.concatenate((pulse,entropy,freq,entreegs,minmaxvarray,MMD,PFD,fmax,breath,RRtest),axis=1)
    return np.concatenate((pulse,entropy,freq,entreegs,minmaxvarray,MMD,PFD,fmax,breath),axis=1)

