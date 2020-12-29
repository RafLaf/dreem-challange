import sys
import importlib
import pandas as pd
from load_features import *

modelname = 'forest'

# Read args
if len(sys.argv) > 1:
    modelname = sys.argv[1]

model_module = importlib.import_module("src.models." + modelname)
model = model_module.gen_model(n_estimators=2000, min_samples_split=2, min_samples_leaf=2, max_features="auto", max_depth=60, criterion="gini", bootstrap=False)

Xtrain,Xtest=read('train'),read('test')
Xtrain,Xtest=scale(Xtrain,Xtest)
y=readlabel()

model.fit(Xtrain,y)
pred=model.predict(Xtest)
pd.DataFrame(pred).to_csv(f"../data/predictions/submission_{modelname}.csv")
