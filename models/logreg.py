import gc

import mne
import numpy as np

from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression


## TODO build a pipeline to get X and y with the desired filter and stuff, move this stuff to another file
raw = mne.io.read_raw("../data/mne/X_train_raw.fif", preload=True)
raw.filter(.5, 25, fir_design='firwin')

events = mne.find_events(raw, stim_channel='sleep_state', initial_event=True)
event_id = dict(awake=1, state_1=2, state_2=3, SWS=4, REM=5)

# TODO Try picking ecg and ecg too
picks = mne.pick_types(raw.info, meg=False, eeg=True, ecg=False, misc=False, stim=False, exclude='bads')

# TODO Reject the right values
reject = dict(eeg=1.)

epochs = mne.Epochs(raw, events, event_id, 0., 6., proj=False, picks=picks, baseline=None, reject=reject)
## 

del raw
gc.collect()

X = epochs.get_data()
y = epochs.events[:, 2]

del epochs
gc.collect()

dim = X.shape
X = X.reshape([dim[0], dim[1] * dim[2]])
X = (X - X.mean()) / X.std()


kf = KFold(n_splits=4)
kf.get_n_splits(X)

scores = []
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    logreg = LogisticRegression(solver='lbfgs')
    logreg.fit(X_train, y_train)

    pred = logreg.predict(X_test)
    score = f1_score(y_test, pred, average='weighted')

    print(score)
    scores.append(score)

    ## TODO: Find out why it is not converging, maybe try other solvers or preprocessing
