import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy import stats

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

def read(mode='train', RC=False, PATH=PATH, rm_outliers=False):
    feature_list = ["lpulse", "entropy", "fbands", "entreeg", "minmaxvarray", "MMD", "fmax", "PFD", "breath", "agitation"]
    if RC:
        feature_list.append("bestfeatR")

    features = []
    for f in feature_list:
        features.append(np.load(f'{PATH}/interim/{f}{mode}.npy', allow_pickle = True))

    features = np.concatenate(features, axis=1)

    if rm_outliers == False:
        return features

    idx = (np.abs(stats.zscore(features)) < 3).all(axis=1)
    return features[idx], idx
