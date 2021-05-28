from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from numpy import array

#1. 데이터
x = np.array(range(1,101))
#x2 = array(range(1,101))
y = np.array(range(101,201))

x_train = x[:60]
x_al = x[60:80]
x_test = x[80:]

y_train = y[:60]
y_val = y[60:80]
y_test = y[80:]

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
print('loss : ', loss)

result = model.predict([101,102,103])
print('result : ', result)