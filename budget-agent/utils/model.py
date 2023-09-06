#!/usr/bin/env python
# model.py

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

import math
from sklearn.metrics import mean_squared_error

__author__ = 'jhsiao'

def model_train(X_train, y_train):
    # Initializing the RNN
    regressor = Sequential()

    # Adding the first LSTM layer and some dropout reg
    # w/ dropout rate: p
    regressor.add(
        LSTM(
            units=50,
            return_sequences=True,
            input_shape=(X_train.shape[1], X_train.shape[2])
        )
    )
    p = 0.2
    regressor.add(Dropout(p))

    # Adding a 2nd LSTM layer and some dropout reg
    regressor.add(
        LSTM(
            units=50,
            return_sequences=True,
        )
    )
    regressor.add(Dropout(p))

    # Adding a 3rd LSTM layer and some dropout reg
    regressor.add(
        LSTM(
            units=50,
            return_sequences=True,
        )
    )
    regressor.add(Dropout(p))

    # Adding a 4th LSTM layer and some dropout reg
    regressor.add(
        LSTM(
            units=50,
            return_sequences=False,
        )
    )
    regressor.add(Dropout(p))

    # Adding the output layer
    regressor.add(Dense(units=1))

    # Compiling the RNN
    regressor.compile(optimizer='adam', loss='mean_squared_error')

    # Fitting the RNN to the training set
    regressor.fit(X_train, y_train, epochs=100, batch_size=8)
    return regressor

def model_test(X_test, regressor):
    predicted_spd = regressor.predict(X_test)
    return predicted_spd
