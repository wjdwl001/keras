from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from numpy import array

#1. 데이터
x = np.array(range(1,101))
#x2 = array(range(1,101))
y = np.array(range(101,201))
from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.4, train_size=0.6
    #, shuffle=True, train_size=0.8
)
x_test, x_val, y_test, y_val = train_test_split(
    x_test, y_test, train_size=0.5
    #, shuffle=True, train_size=0.8
)
print(x_train)
print(x_val)
print(x_test)
print(x_train.shape)
print(x_val.shape)
print(x_test.shape)

#2. 모델 구성
model = Sequential()
model.add(Dense(5, activation='relu', input_dim=1))
model.add(Dense(3))
model.add(Dense(4))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(optimizer="adam", loss='mse')
model.fit(x_train, y_train, epochs=1000)

#4. 평가, 예측
y_predict = model.predict([101,102,103])
print('y_predict : ',y_predict)


