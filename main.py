
from entropy import *
import numpy as np
import matplotlib.pyplot as plt
from load import load_data

# load train data: 7 eeg channels, one oximeter and x y z of head movements
eeg1, eeg2, eeg3, eeg4, eeg5, eeg6, eeg7, pulse, x, y, z = load_data(
        "/Users/fanyang/Documents/dreem-challange/data/X_train.h5")

y_train = pd.read_csv("/Users/fanyang/Documents/dreem-challange/data/y_train.csv")

plt.plot(eeg1[468])

## compute FFT

fs = 250          # Sampling rate (250 Hz)
data = eeg1[468]  # 6 sec of data

# Get real amplitudes of FFT (only in postive frequencies)
fft_vals = np.absolute(np.fft.rfft(data))

# Get frequencies for amplitudes in Hz
fft_freq = np.fft.rfftfreq(len(data), 1.0/fs)

# Define EEG bands
eeg_bands = {'Delta': (0.5, 4),
             'Theta': (4, 8),
             'Alpha': (8, 12),
             'Beta1': (12, 35)}

# Take the mean of the fft amplitude for each EEG band
eeg_band_fft = dict()
for band in eeg_bands:
    freq_ix = np.where((fft_freq >= eeg_bands[band][0]) &
                       (fft_freq <= eeg_bands[band][1]))[0]
    eeg_band_fft[band] = np.mean(fft_vals[freq_ix])

# Plot the data (using pandas here cause it's easy)
df = pd.DataFrame(columns=['band', 'val'])
df['band'] = eeg_bands.keys()
df['val'] = [eeg_band_fft[band] for band in eeg_bands]
ax = df.plot.bar(x='band', y='val', legend=False)
ax.set_xlabel("EEG band")
ax.set_ylabel("Mean band Amplitude")

# Entropy
print(perm_entropy(data, order=3, normalize=True))                 # Permutation entropy
print(spectral_entropy(data, 100, method='welch', normalize=True)) # Spectral entropy
print(svd_entropy(data, order=3, delay=1, normalize=True))         # Singular value decomposition entropy
print(app_entropy(data, order=2, metric='chebyshev'))              # Approximate entropy
print(sample_entropy(data, order=2, metric='chebyshev'))           # Sample entropy
print(lziv_complexity('01111000011001', normalize=True))        # Lempel-Ziv complexity

# Fractal dimension
print(petrosian_fd(data))            # Petrosian fractal dimension
print(katz_fd(data))                 # Katz fractal dimension
print(higuchi_fd(data, kmax=10))     # Higuchi fractal dimension
print(detrended_fluctuation(data))   # Detrended fluctuation analysis

