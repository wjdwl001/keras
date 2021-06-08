import numpy as np
from sklearn.datasets import load_boston

#1. 데이터
dataset = load_boston()
x = dataset.data
y = dataset.target  #label
print(x[:5])
print(y[:5])
print(x.shape, y.shape) #(506,13)(506,) => x 13개 y 1개짜리 모델
print(dataset.feature_names) #'CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 
                             #'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO' 'B' 'LSTAT'
print(dataset.DESCR)

"""                             
    CRIM ZN INDUS CHAS NOX RM AGE DIS RAD TAX PTRATIO B LSTAT | y
1  |                                                          |
...|                                                          |
506|                                                          |
"""

from sklearn.model_selection import train_test_split

#2. 모델 구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense,Input

input1 = Input(shape=(13,))
dense1 = Dense(3)(input1) #상단 레이어의 이름
dense2= Dense(3)(dense1)
output1 = Dense(1)(dense2)

modelF = Model(inputs = input1, outputs = output1)
modelF.summary()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.8, random_state=66
)
print(x_train.shape) #(80,5)
print(y_train.shape) #(80,2)

#3. 컴파일 훈련
modelF.compile(loss='mse',optimizer='adam',metrics=['mae'])
modelF.fit(x_train,y_train, epochs=100, batch_size=1,
          verbose=1) 


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
