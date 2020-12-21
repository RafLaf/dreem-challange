"""
# Usage: 
1. Your PYTHONPATH has to be set to PYTHONPATH:pathtodreemdirectory
2. Go from the terminal in the directory of this file
3. Execute python eval.py `modelname` `datatype`

# Example:
`python eval.py logreg epochs`
"""


import gc
import sys
import importlib

import mne
import numpy as np

from sklearn.metrics import f1_score
from sklearn.model_selection import KFold

modelname = 'cnn'
dataname = 'epochs'

# Read args
if len(sys.argv) > 1:
    modelname = sys.argv[1]

if len(sys.argv) > 2:
    data = sys.argv[2]

# Select model and data
model_module = importlib.import_module("src.models." + modelname)

data_module = importlib.import_module("utils.load_" + dataname)
X, y = data_module.load(reshape=False)

kf = KFold(n_splits=4)
kf.get_n_splits(X)

scores = []
for train, test in kf.split(X):
    model = model_module.gen_model()
    model.fit(X[train], y[train], epochs=1)

    pred = model.predict(X[test])
    score = f1_score(y[test], pred, average='weighted')

    print(f"Score: {round(np.mean(score), 4)}")
    scores.append(score)

print(f"Mean score: {round(np.mean(scores), 4)}")
