import pandas as pd
import numpy as np
import h5py


def load_data(filename = "data/X_train.h5"):
    eeg1 = pd.DataFrame(np.array(h5py.File(filename)['eeg_1'])).apply(lambda x: x / 1e8)
    eeg2 = pd.DataFrame(np.array(h5py.File(filename)['eeg_2'])).apply(lambda x: x / 1e8)
    eeg3 = pd.DataFrame(np.array(h5py.File(filename)['eeg_3'])).apply(lambda x: x / 1e8)
    eeg4 = pd.DataFrame(np.array(h5py.File(filename)['eeg_4'])).apply(lambda x: x / 1e8)
    eeg5 = pd.DataFrame(np.array(h5py.File(filename)['eeg_5'])).apply(lambda x: x / 1e8)
    eeg6 = pd.DataFrame(np.array(h5py.File(filename)['eeg_6'])).apply(lambda x: x / 1e8)
    eeg7 = pd.DataFrame(np.array(h5py.File(filename)['eeg_7'])).apply(lambda x: x / 1e8)
    # index = pd.DataFrame(np.array(h5py.File(filename)['index']))
    # indexwindow = pd.DataFrame(np.array(h5py.File(filename)['index_window']))
    # indexabsolute = pd.DataFrame(np.array(h5py.File(filename)['index_absolute']))
    pulse = pd.DataFrame(np.array(h5py.File(filename)['pulse'])).apply(lambda x: x / 1e10)
    x = pd.DataFrame(np.array(h5py.File(filename)['x'])).apply(lambda x: x / 1e1)
    y = pd.DataFrame(np.array(h5py.File(filename)['y'])).apply(lambda x: x / 1e1)
    z = pd.DataFrame(np.array(h5py.File(filename)['z'])).apply(lambda x: x / 1e1)

    return eeg1, eeg2, eeg3, eeg4, eeg5, eeg6, eeg7, pulse, x, y, z

# eeg1, eeg2, eeg3, eeg4, eeg5, eeg6, eeg7, pulse, x, y, z = load_data()
