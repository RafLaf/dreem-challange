import sys
import importlib
import pandas as pd
from load_features import *

modelname = 'svm'

# Read args
if len(sys.argv) > 1:
    modelname = sys.argv[1]

model_module = importlib.import_module("src.models." + modelname)
model = model_module.gen_model()

Xtrain,Xtest=read('train'),read('test')
Xtrain,Xtest=scale(Xtrain,Xtest)
y=readlabel()

model.fit(Xtrain,y)
pred=model.predict(Xtest)
pd.DataFrame(pred).to_csv(f"../data/predictions/submission_{modelname}.csv")
