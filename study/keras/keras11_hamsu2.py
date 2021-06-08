#다:1 함수형 모델

import numpy as np
#1. 데이터
x = np.array([range(100),range(301,401),range(1,101),
              range(100),range(301,401) ])
y = np.array(range(711,811))
print(x.shape) #(5,100)
print(y.shape) #(2,100)

x=np.transpose(x) #(100,5)
y=np.transpose(y) #(100,2)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.8, random_state=66
)
print(x_train.shape) #(80,5)
print(y_train.shape) #(80,2)

#2. 모델구성
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense,Input
#modelS = Sequential()
#modelS.add(Dense(3, input_shape=(5,)))
#modelS.add(Dense(4))
#modelS.add(Dense(1))
#modelS.summary()

input1 = Input(shape=(5,))
dense1 = Dense(3)(input1) #상단 레이어의 이름
dense1_1 = Dense(7)(dense1)
dense2 = Dense(4)(dense1_1)
output1 = Dense(1)(dense2)

modelF = Model(inputs = input1, outputs = output1)
modelF.summary()


#3. 컴파일 훈련
modelF.compile(loss='mse',optimizer='adam',metrics=['acc'])
modelF.fit(x_train,y_train, epochs=100, batch_size=1,
          verbose=1) 

"""
verbose = 0 : 안나옴
verbose = 1 : 디폴트
verbose = 2 : 프로그래스바 x
verbose = 3 : 
"""

#4. 평가 예측
results = modelF.evaluate(x_test,y_test)
print("results : ",results)


y_pred = modelF.predict(x_test)

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict) : 
    return np.sqrt(mean_squared_error(y_test,y_predict))
print("RMSE : ", RMSE(y_test, y_pred))
print("mse : ", mean_squared_error(y_test, y_pred))

from sklearn.metrics import r2_score
R2 = r2_score(y_test,y_pred)
print("R2 : ", R2)
