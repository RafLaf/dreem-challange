import sys
import yaml
import importlib
import pandas as pd
from utils.load_features import *

modelname = 'xgb'

# Read args
if len(sys.argv) > 1:
    modelname = sys.argv[1]

with open(f"params/{modelname}.yml") as f:
    params = yaml.load(f, Loader=yaml.FullLoader)

model_module = importlib.import_module("src.models." + modelname)
model = model_module.gen_model(**params)

    
Xtrain, idx = read('train', RC=True, rm_outliers=True)
Xtest = read('test', RC=True)
Xtrain, Xtest = scale(Xtrain, Xtest)
y=readlabel()
y = y[idx]

model.fit(Xtrain,y)
pred=model.predict(Xtest)
pd.DataFrame(pred).to_csv(f"../data/predictions/submission_{modelname}.csv")
