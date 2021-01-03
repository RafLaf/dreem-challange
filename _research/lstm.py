import gc
from numpy import around

# import tensorflow as tf
# TODO from tensorflow import keras
from keras.models import Sequential
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Activation, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from utils import load_epochs

X, y = load_epochs.load(reshape=False)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

del X, y
gc.collect()

model = Sequential()
model.add(LSTM(input_shape = (window_size, 1), 
              units = window_size, 
              return_sequences = True))
model.add(Dropout(0.5))
model.add(LSTM(256))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation("linear"))
model.compile(loss = "mse", 
              optimizer = "adam")
model2.summary()








model.fit(X_train, y_train, epochs=100, batch_size=1, verbose=2)

pred = model.predict(X_test)

f1_score(y_test, around(pred), average='weighted')

model.save('/home/ginko/dreem/data/models/lstm1')
# model = keras.models.load_model('path/to/location')

