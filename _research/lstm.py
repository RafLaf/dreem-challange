import gc
from numpy import around

# import tensorflow as tf
# TODO from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from utils import load_epochs

X, y = load_epochs.load(reshape=False)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

del X, y
gc.collect()

look_back = 1501
# create and fit the LSTM network
model = Sequential()
model.add(LSTM(1, input_shape=(7, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, y_train, epochs=100, batch_size=1, verbose=2)

pred = model.predict(X_test)

f1_score(y_test, around(pred), average='weighted')

model.save('/home/ginko/dreem/data/models/lstm1')
# model = keras.models.load_model('path/to/location')

