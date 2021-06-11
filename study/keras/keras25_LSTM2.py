import numpy as np

x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6]])
y = np.array([4,5,6,7])
print(x.shape) #(4,3) -> LSTM으로 사용하기 위해 3차원으로 만든뒤 행 무시 -> input_shape : (3,1)
print(y.shape) 

x = x.reshape(4,3,1)
print(x.shape)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM

model = Sequential()
model.add(LSTM(10,input_shape=(3,1)))
model.add(Dense(10))
model.add(Dense(1))

#model.summary()

#3. 컴파일, 훈련
model.compile(optimizer='adam',loss='mse')
model.fit(x,y,epochs=100,batch_size=1)

#4. 평가, 예측
results = model.evaluate(x,y)
print('loss : ', results)

x_pred = np.array([5,6,7]) #(3,)-> (1,3,1) : [[[5],[6],[7]]]
x_pred = x_pred.reshape(1,3,1)

y_pred = model.predict(x_pred)
print(y_pred)