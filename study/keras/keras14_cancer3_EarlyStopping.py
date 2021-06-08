# Binary Classification

import numpy as np
from sklearn.datasets import load_breast_cancer

#1. Data
dataset = load_breast_cancer()
x = dataset.data
y = dataset.target

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.8, random_state=66
)
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, shuffle=True, train_size=0.75, random_state=66
)

#2. Model
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input

model = Sequential()
model.add(Dense(256, input_shape=(30,), activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(2, activation='softmax'))

#3. Compile, Train
from tensorflow.keras.callbacks import EarlyStopping
early_stopper = EarlyStopping(monitor='val_loss', patience=10, mode='min')

model.compile(
    loss='categorical_crossentropy', optimizer='adam', metrics=['acc']
)

model.fit(
    x_train, y_train, batch_size=1, epochs=64, 
    verbose=2, validation_data=(x_val, y_val),
    callbacks=[early_stopper]
)

#4. Evaluate, Predict
print(model.evaluate(x_test,y_test))
print(model.predict(x_test[:5]))
print(y_test[:5])