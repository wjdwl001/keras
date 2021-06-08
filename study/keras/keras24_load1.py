import numpy as np
from tensorflow.keras.datasets import mnist

#1. 데이터
(x_train,y_train),(x_test,y_test) = mnist.load_data()

print(x_train.shape, x_test.shape) #(60000, 28, 28) (10000, 28, 28)
print(y_train.shape, y_test.shape) #(60000,) (10000,)

#데이터 전처리 및 정규화
x_train = x_train.reshape(60000,28,28,1).astype('float32')/255.
x_test = x_test.reshape(10000,28,28,1).astype('float32')/255.

#원핫벡터 변환
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)


#2. 모델
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense,Conv2D, Flatten

model = load_model('k23_1_model_1.h5')
model.summary()

'''
model = Sequential()
model.add(Conv2D(filters=100, kernel_size=(2,2), padding="same",
                strides=1,input_shape=(28,28,1)))
model.add(Conv2D(20,(2,2)))
model.add(Flatten())
model.add(Dense(100,activation='relu'))
model.add(Dense(10,activation='softmax'))

model.save('./keras/Model/k23_1_model_1.h5')
'''

#3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['acc'])
model.fit(x_train,y_train,epochs=10)

#4. 평가, 예측
results = model.evaluate(x_test,y_test)
print("loss : ",results[0])
print("acc : ",results[1])