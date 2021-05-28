import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# x = np.array([1,2,3,4,5,6,7,8,9,10])
x = np.arange(1,11)
y = np.array([1,2,4,3,5,5,7,9,8,11])


#2. Model
model = Sequential()
model.add(Dense(1, input_shape=(1,)))

#3. Compile
model.compile(loss='mse', optimizer='adam')
model.fit(x,y,epochs=100)

#4. Eval, Predict
y_pred = model.predict(x)
print(y)
print(y_pred)

import matplotlib.pyplot as plt
plt.scatter(x,y)
plt.show()