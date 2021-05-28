#과제
#R2를 음수가 아닌 0.5 이하로 만들 것
#1. 레이어는 인풋과 아웃풋을 포함해서 6개 이상
#2. batch_size = 1
#3. epochs = 100 이상
#4. 히든레이어의 노드의 개수는 10이상 1000이하, 깊이는 4이상
#5. 데이터 조작 금지

import numpy as np

#1. 데이터
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])
x_test = np.array([11,12,13,14,15])
y_test = np.array([11,12,13,14,15])
x_pred = np.array([16,17,18])
#x_test로 y_pred를 만든다.

#2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(30,input_shape=(1,)))
model.add(Dense(15))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일 훈련
model.compile(optimizer='adam', loss='mse',metrics=['acc'])
model.fit(x_train, y_train, epochs=10,batch_size=1)

#4. 평가, 예측
results = model.evaluate(x_test,y_test)
print("results [mse, accuracy]: ", results)

#x_pred = np.array([[11,12,13],[21,22,23]])
#x_pred = np.transpose(x_pred)
#y_pred = model.predict(x_pred)
#print(y_pred)

y_pred = model.predict(x_test)

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict) : 
    return np.sqrt(mean_squared_error(y_test,y_predict))
print("RMSE : ", RMSE(y_test, y_pred))
print("mse : ", mean_squared_error(y_test, y_pred))

from sklearn.metrics import r2_score
R2 = r2_score(y,y_pred)
print("R2 : ", R2)

