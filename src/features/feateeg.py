import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy.stats as scs
#first change paths lines 12 and 17
#first run eggs,mode=read(mode='train' or 'test')
#then extract the different features



pathtosave="/home/raphael/Documents/dreem-challange-main/data/interim/"

def read(mode='train'):
    #mode = 'test'

    filename = "/media/raphael/Data/Dataaccess/ML_DREEM/X_"+str(mode)+"/X_"+str(mode)+".h5"
    #filename = "data/raw/X_"+str(mode)+"/X_"+str(mode)+".h5"
    eggs = []
    for i in range(1, 8):
        np_array = np.array(h5py.File(filename, mode='r')['eeg_'+str(i)])
        eggs.append(np_array)
    eggs=np.array(eggs)
    return eggs,mode


def extractall(save=True):
    a=extractfbandsfull(save=save)[0]
    b=extractentropeeg(save=save)[0]
    c=extractvar(save=save)[0]
    d=extractMMD(save=save)[0]
    e=extractfreqmax(save=save)[0]
    return a,b,c,d,e,mode

def extractfbandsfull(save=True,cuts=1):
    L=list()
    for i in range(7):
        L.append(extractfbands(eggs[i],cuts))
    L=np.array(L)
    s=L.shape
    L=L.reshape((s[0],s[1],s[2],s[3]))
    L=np.moveaxis(L,[3],[0])
    s=L.shape
    print(s)
    L=np.reshape(L,(s[0],L[0].size))
    if save==True:
        np.save(pathtosave+"fbands"+str(mode)+".npy", L)
    return L,mode


def extractentropeeg(save=True,bins=1000):
    entreegs=np.array([entropyarray(eggs[i],bins) for i in range(7)])
    if save==True:
        np.save(pathtosave+'entreeg'+str(mode)+'.npy',entreegs.T)
    return entreegs.T,mode



def extractvar(save=True):
    varray=np.array([np.log(np.var(eggs[i],axis=1)) for i in range(7)])
    varray=varray
    minmax=np.array([(np.max(eggs[i],axis=1)-np.min(eggs[i],axis=1)) for i in range(7)])
    minmaxvarray=np.array([minmax.T,varray.T])    
    minmaxvarray= np.moveaxis(minmaxvarray,[1],[0])
    s=minmaxvarray.shape
    minmaxvarray=minmaxvarray.reshape((s[0],s[1]*s[2]))
    if save==True:
        np.save(pathtosave+'minmaxvarray'+str(mode)+'.npy',minmaxvarray)
    return minmaxvarray,mode



def extractMMD(save=True):
    MMD=np.array([np.log(partitionsum(eggs[i],3)) for i in range(7)])
    if save==True:
        np.save(pathtosave+'MMD'+str(mode)+'.npy',MMD.T)
    return MMD.T,mode



def extractfreqmax(save=True,cuts=1):
    fmax=np.array([extractfmax(eggs[i],cuts) for i in range(7)]).T
    if save==True:
        np.save(pathtosave+'fmax'+str(mode)+'.npy',fmax)
    return fmax,mode


#--------------------------------------------------------------------------------------------#



def d(epochs):
    '''epochs an array of time series cut in time the way you want
    output is a distance d
    '''
    dx=abs(np.argmax(epochs,axis=1)-np.argmin(epochs,axis=1)) 
    dy=np.max(epochs,axis=1)-np.min(epochs,axis=1)
    return np.sqrt(dx**2+dy**2)
    
def partitionsum(ts,lepoch=10,lts=30):
    '''ts : array of time series
    lepoch :duration of epoch
    lts = duration of time series
    '''
    if lts%lepoch!=0:
        print('lts must be a multiple of lepoch')
        
    indexts=ts.shape[1]    
    nepoch=int(lts/lepoch)
    indexepoch=int(indexts/nepoch)
    sum=0
    for i in range (nepoch):
        sum+=d(ts[:,i*indexepoch:(i+1)*indexepoch])
    return sum





def entropyarray(x,bins):
    L=list()
    for elt in x:
        L.append(entropySignal(elt,bins))
    return np.array(L)


def entropySignal(x,bins):
    hist=np.histogram(x,bins=bins,density=True)[0]
    return scs.entropy(hist)

def extractfbands(eeg,cuts):
    L=list()
    s=eeg.shape
    idcut=int(s[1]/cuts)
    freq=np.fft.fftfreq(idcut,1/50)[:int(idcut/2)]
    plt.plot(freq)
    for i in range(cuts):
        print(i)
        FFT=np.fft.fft(eeg[:,i*idcut:(i+1)*idcut],axis=1)[:,:int(idcut/2)]
        FFT=abs(FFT)
        delta=np.sum(FFT[:,np.where((freq<4)*(freq>0))],axis=2)
        theta=np.sum(FFT[:,np.where((freq>4)* (freq<8))],axis=2)
        alpha=np.sum(FFT[:,np.where((freq>8)* (freq<13))],axis=2)
        beta=np.sum(FFT[:,np.where((freq>13 )* (freq<22))],axis=2)
        gamma=np.sum(FFT[:,np.where(freq>22)],axis=2)
        L.append( np.array([delta,theta,alpha,beta,gamma]))
    return np.array(L)


def extractfmax(eeg,cuts):
    L=list()
    s=eeg.shape
    idcut=int(s[1]/cuts)
    freq=np.fft.fftfreq(idcut,1/50)[:int(idcut/2)]
    for i in range(cuts):
        FFT=np.fft.fft(eeg[:,i*idcut:(i+1)*idcut],axis=1)[:,:int(idcut/2)]
        FFT=abs(FFT)
        f=np.sum(FFT*freq,axis=1)/np.sum(FFT,axis=1)
        L.append(f)
    return np.array(L)


