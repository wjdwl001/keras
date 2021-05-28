import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([[10,85,70],[90,85,100],
              [80,50,30],[43,60,100]]) #(4, 3) 모의고사 1-4회차 [국,영,수]
y = np.array([75,65,33,80]) #(3, ) 체육 모의고사 1-4회차 [1,2,3,4회차]

#2. 모델 구성
model = Sequential()
model.add(Dense(10,input_shape=(3,)))
model.add(Dense(5))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일
model.compile(optimizer='adam',loss="mse")
model.fit(x,y,epochs=1000)

#4. 평가, 예측
loss = model.evaluate(x,y)
print("loss : ", loss)
