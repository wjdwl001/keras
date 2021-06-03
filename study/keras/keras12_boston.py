import numpy as np
from sklearn.datasets import load_boston

#1. 데이터
dataset = load_boston()
x = dataset.data
y = dataset.target  #label
print(x[:5])
print(y[:5])
print(x.shape, y.shape)