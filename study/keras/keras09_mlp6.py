#다:1 mlp

import numpy as np
#1. 데이터
x = np.array([range(100),range(301,401),range(1,101),
              range(100),range(301,401) ])
y = np.array([range(711,811), range(1,101)])
print(x.shape) #(5,100)
print(y.shape) #(2,100)

#    x1 x2 x3 x4 x5 y1 y2
#1   
#..
#100  

x=np.transpose(x) #(100,5)
y=np.transpose(y) #(100,2)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.8, random_state=66
)
print(x_train.shape) #(80,5)
print(y_train.shape) #(80,2)

#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
model = Sequential()
model.add(Dense(5, input_shape=(5,)))
model.add(Dense(2))

#3. 컴파일 훈련
model.compile(loss='mse',optimizer='adam',metrics=['acc'])
model.fit(x_train,y_train, epochs=100, batch_size=1)

#4. 평가 예측
results = model.evaluate(x_test,y_test)
print("results : ",results)

y_pred = model.predict(x_test)

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict) : 
    return np.sqrt(mean_squared_error(y_test,y_predict))
print("RMSE : ", RMSE(y_test, y_pred))
print("mse : ", mean_squared_error(y_test, y_pred))

from sklearn.metrics import r2_score
R2 = r2_score(y_test,y_pred)
print("R2 : ", R2)