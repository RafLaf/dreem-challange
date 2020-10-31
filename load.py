import pandas as pd
import numpy as np
import h5py


def load_data(filename = "data/X_train.h5"):
    eeg1 = pd.DataFrame(np.array(h5py.File(filename)['eeg_1']))
    eeg2 = pd.DataFrame(np.array(h5py.File(filename)['eeg_2']))
    eeg3 = pd.DataFrame(np.array(h5py.File(filename)['eeg_3']))
    eeg4 = pd.DataFrame(np.array(h5py.File(filename)['eeg_4']))
    eeg5 = pd.DataFrame(np.array(h5py.File(filename)['eeg_5']))
    eeg6 = pd.DataFrame(np.array(h5py.File(filename)['eeg_6']))
    eeg7 = pd.DataFrame(np.array(h5py.File(filename)['eeg_7']))
    # index = pd.DataFrame(np.array(h5py.File(filename)['index']))
    # indexwindow = pd.DataFrame(np.array(h5py.File(filename)['index_window']))
    # indexabsolute = pd.DataFrame(np.array(h5py.File(filename)['index_absolute']))
    pulse = pd.DataFrame(np.array(h5py.File(filename)['pulse']))
    x = pd.DataFrame(np.array(h5py.File(filename)['x']))
    y = pd.DataFrame(np.array(h5py.File(filename)['y']))
    z = pd.DataFrame(np.array(h5py.File(filename)['z']))

    return eeg1, eeg2, eeg3, eeg4, eeg5, eeg6, eeg7, pulse, x, y, z

# eeg1, eeg2, eeg3, eeg4, eeg5, eeg6, eeg7, pulse, x, y, z = load_data()
