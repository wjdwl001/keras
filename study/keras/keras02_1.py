import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x_train = np.array([1,2,3,4,5])
y_train = np.array([1,2,3,4,5])
x_test = np.array([6,7,8])
y_test = np.array([6,7,8])

#2. 모델 구성
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(3))
model.add(Dense(4))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(optimizer="adam", loss='mse')
model.fit(x_train, y_train, epochs=1000)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
predict = model.predict([9])
print("loss = ", loss)
print("predict = ", predict)