import numpy as np
import h5py


def load_data(filename = "data/X_train.h5"):
    eeg1 = np.array(h5py.File(filename, "r")['eeg_1'])
    eeg2 = np.array(h5py.File(filename, "r")['eeg_2'])
    eeg3 = np.array(h5py.File(filename, "r")['eeg_3'])
    eeg4 = np.array(h5py.File(filename, "r")['eeg_4'])
    eeg5 = np.array(h5py.File(filename, "r")['eeg_5'])
    eeg6 = np.array(h5py.File(filename, "r")['eeg_6'])
    eeg7 = np.array(h5py.File(filename, "r")['eeg_7'])
    # index = pd.DataFrame(np.array(h5py.File(filename)['index']))
    # indexwindow = pd.DataFrame(np.array(h5py.File(filename)['index_window']))
    # indexabsolute = pd.DataFrame(np.array(h5py.File(filename)['index_absolute']))
    pulse = np.array(h5py.File(filename, "r")['pulse'])
    x = np.array(h5py.File(filename, "r")['x'])
    y = np.array(h5py.File(filename, "r")['y'])
    z = np.array(h5py.File(filename, "r")['z'])

    return eeg1, eeg2, eeg3, eeg4, eeg5, eeg6, eeg7, pulse, x, y, z

# eeg1, eeg2, eeg3, eeg4, eeg5, eeg6, eeg7, pulse, x, y, z = load_data()
