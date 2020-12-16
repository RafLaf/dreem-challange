import gc
import sys
import importlib

import mne
import numpy as np

from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression

# Read args
if len(sys.argv) > 1:
    modelname = sys.argv[1]
else:
    modelname = 'logreg'

if len(sys.argv) > 2:
    data = sys.argv[2]
else:
    data = 'epochs'

# Select model and data
module = importlib.import_module("models." + modelname)
model = module.gen_model()

if data == 'epochs':
    epochs = mne.read_epochs("../data/mne/X_train_epo.fif", proj=True)
    X = epochs.get_data()
    y = epochs.events[:, 2]
    del epochs
    gc.collect()
    dim = X.shape
    X = X.reshape([dim[0], dim[1] * dim[2]])
    X = (X - X.mean()) / X.std()

elif data == 'raw':
    pass # TODO add something here when you have models using raw


kf = KFold(n_splits=4)
kf.get_n_splits(X)

scores = []
for train, test in kf.split(X):
    logreg = LogisticRegression(solver='lbfgs')
    logreg.fit(X[train], y[train])

    pred = logreg.predict(X[test])
    score = f1_score(y[test], pred, average='weighted')

    print(f"Score: {round(np.mean(score), 4)}")
    scores.append(score)
    break

print(f"Mean score: {round(np.mean(scores), 4)}")
    ## TODO: Find out why it is not converging, maybe try other solvers or preprocessing

