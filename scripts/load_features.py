import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

PATH="../data"

def scale(Xtrain,Xtest=[]):
    scaler = StandardScaler()
    scaler.fit(Xtrain)
    Xtrain=scaler.transform(Xtrain)
    if len(Xtest) == 0:
        return Xtrain
    Xtest=scaler.transform(Xtest)
    return Xtrain,Xtest

def readlabel(PATH=PATH):
    filename = f"{PATH}/raw/y_train.csv"
    ytrain=np.array(pd.read_csv(filename))
    y=ytrain[:,1]
    return y

def read(mode='train',RC=False, PATH=PATH):
    pulse        = np.load(f'{PATH}/interim/lpulse{mode}.npy',       allow_pickle = True)
    entropy      = np.load(f'{PATH}/interim/entropy{mode}.npy',      allow_pickle = True)
    freq         = np.load(f'{PATH}/interim/fbands{mode}.npy',       allow_pickle = True)
    entreegs     = np.load(f'{PATH}/interim/entreeg{mode}.npy',      allow_pickle = True)
    minmaxvarray = np.load(f'{PATH}/interim/minmaxvarray{mode}.npy', allow_pickle = True)
    MMD          = np.load(f'{PATH}/interim/MMD{mode}.npy',          allow_pickle = True)
    fmax         = np.load(f'{PATH}/interim/fmax{mode}.npy',         allow_pickle = True)
    PFD          = np.load(f'{PATH}/interim/PFD{mode}.npy',          allow_pickle = True)
    breath       = np.load(f'{PATH}/interim/breath{mode}.npy',       allow_pickle = True)
    fmax         = np.reshape(fmax,(fmax.shape[0],fmax[0].size))
    if RC==True:
        if mode=='train':
            RRtrain=np.load(f'{PATH}/interim/bestfeatRtrain.npy',allow_pickle=True)
            return np.concatenate((pulse,entropy,freq,entreegs,minmaxvarray,MMD,PFD,fmax,breath,RRtrain),axis=1)
        RRtest=np.load(f'{PATH}/interim/bestfeatRest.npy',allow_pickle=True)
        return np.concatenate((pulse,entropy,freq,entreegs,minmaxvarray,MMD,PFD,fmax,breath,RRtest),axis=1)
    return np.concatenate((pulse,entropy,freq,entreegs,minmaxvarray,MMD,PFD,fmax,breath),axis=1)
