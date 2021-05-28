import numpy as np
#1. 데이터
x = np.array([[1,2,3,4,5,6,7,8,9,10],
             [11,12,13,14,15,16,17,18,19,20]])

y = np.array([1,2,3,4,5,6,7,8,9,10])

print(x.shape) #()
print(y.shape)

x = np.transpose(x)
print(x.shape)


#2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(10, input_dim=2))
model.add(Dense(10,input_shape=(2,)))
model.add(Dense(5))
model.add(Dense(1))

#3. 컴파일
model.compile(optimizer='adam', loss='mse')
model.fit(x, y, epochs=1000)

#4. 평가, 예측
y_predict = model.predict([[11,12,13],[21,22,23]])
print('y_predict : ',y_predict)
