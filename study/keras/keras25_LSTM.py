import numpy as np

x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6]])
y = np.array([4,5,6,7])
print(x.shape)
print(y.shape)

x = x.reshape(4,3,1)
print(x.shape)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM

model = Sequential()
model.add(LSTM(10,input_shape=(3,1)))
model.add(Dense(10))
model.add(Dense(1))

model.summary()