import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy.stats as scs
#first change paths lines 12 and 16

#first run pulse,x,y,z,mode=read(mode='train' or 'test')
#then extract the different features


pathtosave="/home/raphael/Documents/dreem-challange-main/data/interim/"

def read(mode='train'):
    #mode = 'test'
    filename = "/media/raphael/Data/Dataaccess/ML_DREEM/X_"+str(mode)+"/X_"+str(mode)+".h5"
    #filename = "data/raw/X_"+str(mode)+"/X_"+str(mode)+".h5"
    pulse = np.array(h5py.File(filename, mode='r')['pulse'])
    x = np.array(h5py.File(filename, mode='r')['x'])
    y = np.array(h5py.File(filename, mode='r')['y'])
    z =np.array(h5py.File(filename, mode='r')['z'])
    return pulse,x,y,z,mode


def extractall(save=True):
    a=extractpulse(save=save)[0]
    b=extractentropy(save=save)[0]
    return a,b,mode

def extractpulse(save=True):
    Lpulse=[]
    product=pulse[:,1:]*pulse[:,:-1]
    for i in range (product.shape[0]):
        Lpulse.append((np.where(product[i]<0))[0].shape)
    Lpulse=np.array(Lpulse)
    if save==True:
         np.save(pathtosave+'lpulse'+str(mode)+'.npy',Lpulse)
    return Lpulse,mode


def extractentropy(save=True,bins=100):
    entrpulse=entropyarray(pulse,bins)
    entrx=entropyarray(x,bins)
    entry=entropyarray(y,bins)
    entrz=entropyarray(z,bins)
    out= np.transpose(np.array([entrpulse,entrx,entry,entrz]))
    if save==True:
         np.save(pathtosave+'entropy'+str(mode)+'.npy',out)
    return out,mode

#--------------------------------------------------------------------------------------------#


def entropyarray(x,bins):
    L=list()
    for elt in x:
        L.append(entropySignal(elt,bins))
    return np.array(L)

def entropySignal(x,bins):
    hist=np.histogram(x,bins=bins,density=True)[0]
    return scs.entropy(hist)



