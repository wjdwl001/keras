import numpy as np
import tensorflow as tf

#1. 데이터
x = np.array([1,2,3])
y = np.array([1,2,3])

#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(3,input_dim=1)) #입력 개수는 1개
model.add(Dense(4)) #레이어를 추가하면서 노드 개수 정의
model.add(Dense(2))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, batch_size=1 #batch_size : 몇개씩 잘라서 훈련하는지
              , epochs=1000) #epochs : 훈련 횟수

#4. 평가, 예측
loss = model.evaluate(x,y)
print('loss : ',loss)

results = model.predict([4])
print('prediction for x = 4 : ', results)