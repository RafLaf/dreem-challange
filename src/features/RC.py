import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sc
import pandas as pd
import h5py
from sklearn.decomposition import PCA
#first run eggstrain,mode=read(mode='train')
#          eggstest,mode=read(mode='test')
#          eggstrain=scale(eggstrain)
#          eggstest=scale(eggstest)
#          Rtrain,Rtest=extract(save=False)
#          PCARC(Rtrain,Rtest)
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

def scale(eggs):
    for i in range (7):
        eggs[i]-=np.reshape(eggs[0].mean(axis=1),(eggs[0].shape[0],1))
        eggs[i]=np.divide(eggs[i],eggs[i].std(axis=1).reshape((eggs[0].shape[0],1)))
        print(eggs[i].mean(axis=1).shape)
    eggs= np.moveaxis(eggs,[1],[0])
    return eggs

    
def extract(save=True,Nr=5000,start=0,end=-1,step=10,rho=0.99,D=15,nbchan=7,aleak=0.07,gamma=1,biasscale=0.5,snap=4):
    Res=Reservoir(Nr,D,rho,nbchan)
    Rtrain=Res.main(aleak,gamma,biasscale,snap,eggstrain[start:end,:,slice(0,-1,step)])
    Rtest=Res.main(aleak,gamma,biasscale,snap,eggstest[start:end,:,slice(0,-1,step)])
    Rtrain=np.reshape(Rtrain,(Rtrain.shape[0],Rtrain.shape[1]*Rtrain.shape[2]))
    Rtest=np.reshape(Rtest,(Rtest.shape[0],Rtest.shape[1]*Rtest.shape[2]))
    if save==True:
        np.save(pathtosave+'Rtrain.npy',Rtrain)
        np.save(pathtosave+'Rtest.npy',Rtest)
    return Rtrain,Rtest

def PCARC(nbcomp=5,save=True):
    pca=PCA()
    pca.fit(Rtrain)
    RRtrain=pca.transform(Rtrain)
    RRtest=pca.transform(Rtest)
    print('explained variance ratio ',pca.explained_variance_ratio_[:nbcomp])
    print('the next 4 are         : ',pca.explained_variance_ratio_[nbcomp:nbcomp+4])
    if save==True:
        np.save(pathtosave+'RRtrain.npy',RRtrain[:,:nbcomp])
        np.save(pathtosave+'RRtest.npy',RRtest[:,:nbcomp])
    return RRtrain[:,:nbcomp],RRtest[:,:nbcomp]

class Reservoir:
    def __init__(self,Nr,D,rho,nbchan):
        self.Nr=Nr
        self.r=np.zeros(self.Nr)
        self.W=sc.random(self.Nr,self.Nr,density=float(D/self.Nr))
        self.W=rho/max(abs(np.linalg.eigvals(self.W.A)))*self.W
        self.W=(2*self.W-(self.W!=0))
        self.nbchan=nbchan
        self.Win=sc.random(self.Nr,nbchan,density=float(1))
        self.Win=(2*self.Win-(self.Win!=0)).A
    def step(self,input):
        self.r=(1-self.aleak)*self.r+self.aleak*np.tanh(self.bias+self.r@self.W+self.gamma*self.Win@input)
    def snapshots(self,aleak,gamma,biasscale,inputs,snap):
        self.r=np.zeros(self.Nr)
        L=list()
        self.bias=biasscale*2*(np.random.rand(self.Nr))-1
        self.aleak,self.gamma=aleak,gamma
        s=inputs.shape
        if s[1]!=self.nbchan:
          print('nb of channel conflict with inputs dim : reinitialize the reservoir with the right nbchan or change input accodingly')
        indexsnap=int(s[0]/snap)
        for i in range(s[0]):
            self.step(inputs[i])
            if i>0 and i%indexsnap==0:
                L.append(self.r)
        L=np.array(L)
        return L
    def main(self,aleak,gamma,biasscale,snap,sinputs):
        Lmain=list()
        count=0
        for inputs in sinputs:
            count+=1
            if count%100==0:
                print(count)
            self.r=np.zeros(self.Nr)
            Lmain.append(self.snapshots(aleak,gamma,biasscale,inputs.T,snap))
        return np.array(Lmain)


