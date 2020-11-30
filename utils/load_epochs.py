import gc
import mne

def load(reshape=True, normalize=True):
    epochs = mne.read_epochs("../data/mne/X_train_epo.fif", proj=True)

    X = epochs.get_data()
    y = epochs.events[:, 2]

    del epochs
    gc.collect()

    if reshape:
        dim = X.shape
        X = X.reshape([dim[0], dim[1] * dim[2]])

    if normalize:
        X = (X - X.mean()) / X.std()

    return X, y
