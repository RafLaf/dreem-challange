import gc

import pandas as pd
import mne
import numpy as np
from scipy import interpolate

from utils.load import load_data

def sync_interpolate(array):
    x = np.arange(0, len(array), 1)
    f = interpolate.interp1d(x, array)

    xnew = np.arange(0, len(array)-1, 0.1994)
    ynew = f(xnew)
    assert len(ynew) == 1500
    return ynew

def sync_matrix(matrix):
    new_matrix = np.zeros([matrix.shape[0], matrix.shape[1]*5])
    for i in range(matrix.shape[0]):
        new_matrix[i, :] = sync_interpolate(matrix[i, :])
    return new_matrix

def create_mne_raw_object(save=False, proj=True):
    eeg1, eeg2, eeg3, eeg4, eeg5, eeg6, eeg7, pulse, x, y, z = load_data("../data/raw/X_train.h5")
    data = np.zeros([12, np.concatenate(eeg1).shape[0]])


    data[0, :] = np.concatenate(eeg1)
    del eeg1
    data[1, :] = np.concatenate(eeg2)
    del eeg2
    data[2, :] = np.concatenate(eeg3)
    del eeg3
    gc.collect()

    data[3, :] = np.concatenate(eeg4)
    del eeg4
    data[4, :] = np.concatenate(eeg5)
    del eeg5
    data[5, :] = np.concatenate(eeg6)
    del eeg6
    gc.collect()

    data[6, :] = np.concatenate(eeg7)
    del eeg7
    data[7, :] = np.concatenate(sync_matrix(pulse))
    del pulse
    gc.collect()

    data[8, :] = np.concatenate(sync_matrix(x))
    del x
    data[9, :] = np.concatenate(sync_matrix(y))
    del y
    data[10, :] = np.concatenate(sync_matrix(z))
    del z
    gc.collect()

    # get chanel names
    ch_names = ["Fpz", "O1", "F7", "F8", "Fp2", "O2", "Fp1", "pulse", "x", "y", "z", "sleep_state"]
    ch_types = [*['eeg'] * 7, 'ecg', *['misc'] * 3, "stim"]

    # scale
    data *= 1e-7
    data[7] *= 1e-2

    # create and populate MNE info structure
    info = mne.create_info(ch_names, sfreq=250.0, ch_types=ch_types)
    # create raw object
    raw = mne.io.RawArray(data, info, verbose=False)
    raw.set_montage("standard_1020")

    del data
    gc.collect()

    Y = pd.read_csv("../data/raw/y_train.csv")
    Y = np.array(Y.sleep_stage) + 1

    new_events = mne.make_fixed_length_events(raw, start=0.001, stop=148128.5, duration=6.)
    new_events.shape
    new_events[:, 2] = Y
    raw.add_events(new_events, stim_channel ='sleep_state')


    if proj == True:
        projs = mne.compute_proj_raw(raw, n_grad=0, n_mag=0, n_eeg=2, n_jobs=4)
        raw.add_proj(projs)
    if save == True:
        raw.save("../data/mne/X_train_raw.fif")

    return raw

def create_mne_epochs_object(save=False, eeg=True, ecg=False, misc=False, proj=False, rej=0.5):
    raw = mne.io.read_raw("../data/mne/X_train_raw.fif", preload=True)
    raw.filter(.5, 25, fir_design='firwin')

    events = mne.find_events(raw, stim_channel='sleep_state', initial_event=True)
    event_id = dict(awake=1, state_1=2, state_2=3, SWS=4, REM=5)

# TODO Try picking ecg and ecg too
    picks = mne.pick_types(raw.info, meg=False, eeg=eeg, ecg=ecg, misc=misc, stim=False, exclude='bads')

# TODO Reject the right values
    reject = dict(eeg=rej)

    epochs = mne.Epochs(raw, events, event_id, 0., 6., proj=proj, picks=picks, baseline=None, reject=reject)
    del raw
    gc.collect()

    if save == True:
        epochs.save("../data/mne/X_train_epo.fif")
    return epochs
