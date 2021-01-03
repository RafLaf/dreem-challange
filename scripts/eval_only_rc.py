"""
# Usage: 
1. Your PYTHONPATH has to be set to PYTHONPATH:pathtodreemdirectory
2. Go from the terminal in the directory of this file
3. Execute python eval.py `modelname` 

# Example:
`python eval.py logreg`
"""

import sys
import importlib
import yaml
import numpy as np

from sklearn.metrics import f1_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold

from utils.load_features import *

modelname = "xgb"
"""
# Usage: 
1. Your PYTHONPATH has to be set to PYTHONPATH:pathtodreemdirectory
2. Go from the terminal in the directory of this file
3. Execute python eval.py `modelname` 

# Example:
`python eval.py logreg`
"""

import sys
import importlib
import yaml
import numpy as np

from sklearn.metrics import f1_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold

from utils.load_features import *

modelname = "xgb"

if len(sys.argv) > 1:
    modelname = sys.argv[1]

model_module = importlib.import_module("src.models." + modelname)

with open(f"params/{modelname}.yml") as f:
    params = yaml.load(f, Loader=yaml.FullLoader)
    


X = read('train', RC=True)
X = scale(X)
y = readlabel()

X2 = X[:, -20:]
kf = KFold(n_splits=4)
kf.get_n_splits(X2)

scores = []
for train, test in kf.split(X2):
    model2 = model_module.gen_model(**params)
    model2.fit(X2[train], y[train])

    pred = model2.predict(X2[test])
    score = f1_score(y[test], pred, average='weighted')

    print(f"Score: {round(np.mean(score), 4)}")
    scores.append(score)

    print("Confusion Matrix:")
    print(confusion_matrix(y[test], pred))
    break

    print("Classification Report")
    print(classification_report(y[test], pred))

print(f"Mean score: {round(np.mean(scores), 4)}")
# SVM = .644
# RF = .66
# RF with params = .70

