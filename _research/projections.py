import mne
from mne.preprocessing import (create_eog_epochs, create_ecg_epochs, compute_proj_ecg, compute_proj_eog)

raw = mne.io.read_raw("../data/mne/X_train_raw.fif", preload=True)

# projs = mne.compute_proj_raw(raw, n_grad=0, n_mag=0, n_eeg=2, n_jobs=4)
# raw.add_proj(projs)

ecg_evoked = create_ecg_epochs(raw).average()
ecg_evoked.plot_joint()

