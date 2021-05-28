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

#3. 컴파일 훈련
model.compile(optimizer='adam', loss='mse',metrics=['acc'])
model.fit(x, y, epochs=1000,batch_size=1)

#4. 평가, 예측
results = model.evaluate(x,y)
print("results : ", results)

#x_pred = np.array([[11,12,13],[21,22,23]])
#x_pred = np.transpose(x_pred)
#y_pred = model.predict(x_pred)
#print(y_pred)

y_predict = model.predict(x)

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict) : 
    return np.sqrt(mean_squared_error(y_test,y_predict))
print("RMSE : ", RMSE(y, y_predict))
print("mse : ", mean_squared_error(y, y_predict))