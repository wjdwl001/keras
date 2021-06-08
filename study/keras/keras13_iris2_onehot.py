import numpy as np
from sklearn.datasets import load_iris

#1. 데이터
dataset = load_iris()
x = dataset.data
y = dataset.target  #label
print(x[:5])
print(y[:5])
print(x.shape, y.shape) #(150,4)(150,) => x 4개 y 1개짜리 모델


## One-Hot Encoding
from tensorflow.keras.utils import to_categorical

y = to_categorical(y)

#print(dataset.feature_names) #'sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'
#print(dataset.DESCR)

"""                             
    sepal length, sepal width, petal length, petal width (cm) | y
1  |                                                          |
...|                                                          |
150|                                                          |
"""

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.8, random_state=66
)


#2. 모델 구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense,Input

input1 = Input(shape=(4,))
h1 = Dense(10)(input1)
h2 = Dense(10)(h1)
h3 = Dense(10)(h2)
h4 = Dense(5)(h3)
output1 = Dense(3, activation='softmax')(h4)
model = Model(inputs=input1, outputs=output1)


#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
model.fit(x_train,y_train, epochs=100, batch_size=1,
          verbose=1, validation_split=0.1) 


#4. 평가, 예측
results = model.evaluate(x_test,y_test)
print("results : ",results) #loss, acc

y_predict = model.predict(x_test)
print()
print(y_predict[:5])  