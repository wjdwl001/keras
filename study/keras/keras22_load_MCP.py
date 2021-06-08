import numpy as np
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Dense,Conv2D

from sklearn.datasets import load_breast_cancer

#1. Data
dataset = load_breast_cancer()
x = dataset.data
y = dataset.target

#from tensorflow.keras.utils import to_categorical
#y = to_categorical(y)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.8, random_state=66
)

#2. Model
model = load_model('.\keras\CheckPoint\k21_tensor_34-0.07.hdfs')
model.summary()
results = model.evaluate(x_test,y_test)

print('loss : ',results[0])
print('acc : ',results[1])