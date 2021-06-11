import numpy as np
#1. 데이터 
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
              [5,6,7],[6,7,8],[7,8,9],[8,9,10],
              [9,10,11],[10,11,12],
              [20,30,40],[30,40,50],[40,50,60]])
y = np.array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 50, 60, 70])

x_pred = np.array([50,60,70])
x = x.reshape(13,3,1)

#2. 모델 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM

model = Sequential()
model.add(LSTM(10,input_shape=(3,1)))
model.add(Dense(5))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(optimizer='adam',loss='mse')
model.fit(x,y,epochs=10)

#4. 평가, 예측
results = model.evaluate(x,y)
print("loss : ", results)
y_pred = model.predict(x_pred)
print("y predict for ([50,60,70]) : ",y_pred)
