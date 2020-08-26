from sklearn.datasets import load_digits
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import cv2
from sklearn.preprocessing import MinMaxScaler

digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, random_state=66)
# print(digits.data.max)
# a = digits.data[9].reshape(8, 8)
# sc = MinMaxScaler(feature_range=(0, 255))
# a = sc.fit_transform(a)
# cv2.imwrite('./Adaboost/input9.png', a)

# 決定木とAdaBoostRegressorのパラメータ設定
models = {
    'tree': DecisionTreeRegressor(random_state=0),
    'AdaBoost': AdaBoostRegressor(DecisionTreeRegressor(), random_state=0)
}

# モデル構築
scores = {}
for model_name, model in models.items():
    model.fit(X_train, y_train)
    scores[(model_name, 'train_score')] = model.score(X_train, y_train)
    scores[(model_name, 'test_score')] = model.score(X_test, y_test)

# 結果を表示
print(pd.Series(scores).unstack())
print(scores)

input0 = cv2.imread('./img/0.png', 0)
# input0 = cv2.imread('./Adaboost/input0.png', 0)
input0 = cv2.resize(input0, (8, 8))
sc = MinMaxScaler(feature_range=(0, 16))
input0 = sc.fit_transform(input0)
cv2.imwrite('./Adaboost/out0.png', input0)
print(model.predict(input0.reshape(1, 64)))

input1 = cv2.imread('./img/1.png', 0)
# input1 = cv2.imread('./Adaboost/input1.png', 0)
input1 = cv2.resize(input1, (8, 8))
sc = MinMaxScaler(feature_range=(0, 16))
input1 = sc.fit_transform(input1)
cv2.imwrite('./Adaboost/out1.png', input1)
print(model.predict(input1.reshape(1, 64)))

input2 = cv2.imread('./img/2.png', 0)
# input2 = cv2.imread('./Adaboost/input2.png', 0)
input2 = cv2.resize(input2, (8, 8))
sc = MinMaxScaler(feature_range=(0, 16))
input2 = sc.fit_transform(input2)
cv2.imwrite('./Adaboost/out2.png', input2)
print(model.predict(input2.reshape(1, 64)))

input3 = cv2.imread('./img/3.png', 0)
# input3 = cv2.imread('./Adaboost/input3.png', 0)
input3 = cv2.resize(input3, (8, 8))
sc = MinMaxScaler(feature_range=(0, 16))
input3 = sc.fit_transform(input3)
cv2.imwrite('./Adaboost/out3.png', input3)
print(model.predict(input3.reshape(1, 64)))

input4 = cv2.imread('./img/4.png', 0)
# input4 = cv2.imread('./Adaboost/input4.png', 0)
input4 = cv2.resize(input4, (8, 8))
sc = MinMaxScaler(feature_range=(0, 16))
input4 = sc.fit_transform(input4)
cv2.imwrite('./Adaboost/out4.png', input4)
print(model.predict(input4.reshape(1, 64)))

input5 = cv2.imread('./img/5.png', 0)
# input5 = cv2.imread('./Adaboost/input5.png', 0)
input5 = cv2.resize(input5, (8, 8))
sc = MinMaxScaler(feature_range=(0, 16))
input5 = sc.fit_transform(input5)
cv2.imwrite('./Adaboost/out5.png', input5)
print(model.predict(input5.reshape(1, 64)))

input6 = cv2.imread('./img/6.png', 0)
# input6 = cv2.imread('./Adaboost/input6.png', 0)
input6 = cv2.resize(input6, (8, 8))
sc = MinMaxScaler(feature_range=(0, 16))
input6 = sc.fit_transform(input6)
cv2.imwrite('./Adaboost/out6.png', input6)
print(model.predict(input6.reshape(1, 64)))

input7 = cv2.imread('./img/7.png', 0)
# input7 = cv2.imread('./Adaboost/input7.png', 0)
input7 = cv2.resize(input7, (8, 8))
sc = MinMaxScaler(feature_range=(0, 16))
input7 = sc.fit_transform(input7)
cv2.imwrite('./Adaboost/out7.png', input7)
print(model.predict(input7.reshape(1, 64)))

input8 = cv2.imread('./img/8.png',0)
# input8 = cv2.imread('./Adaboost/input8.png', 0)
input8 = cv2.resize(input8, (8, 8))
sc = MinMaxScaler(feature_range=(0, 16))
input8 = sc.fit_transform(input8)
cv2.imwrite('./Adaboost/out8.png', input8)
print(model.predict(input8.reshape(1, 64)))

input9 = cv2.imread('./img/9.png', 0)
# input9 = cv2.imread('./Adaboost/input9.png', 0)
input9 = cv2.resize(input9, (8, 8))
sc = MinMaxScaler(feature_range=(0, 16))
input9 = sc.fit_transform(input9)
cv2.imwrite('./Adaboost/out9.png', input9)
print(model.predict(input9.reshape(1, 64)))
