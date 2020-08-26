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
# a = digits.data[2].reshape(8, 8)
# sc = MinMaxScaler(feature_range=(0, 255))
# a = sc.fit_transform(a)
# cv2.imwrite('./Adaboost/input2.png', a)

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
# input0 = cv2.resize(input0, (8, 8))
# sc = MinMaxScaler(feature_range=(0, 16))
# input0 = sc.fit_transform(input0)
# cv2.imwrite('./Adaboost/out.png', input0)

input1 = cv2.imread('./img/1.png', 0)
# input1 = cv2.resize(input1, (8, 8))
# sc = MinMaxScaler(feature_range=(0, 16))
# input1 = sc.fit_transform(input1)
# cv2.imwrite('./Adaboost/out.png', input1)

input2 = cv2.imread('./img/2.png', 0)
input2 = cv2.resize(input2, (8, 8))
sc = MinMaxScaler(feature_range=(0, 16))
input2 = sc.fit_transform(input2)
cv2.imwrite('./Adaboost/out.png', input2)

input3 = cv2.imread('./img/3.png', 0)
# input3 = cv2.resize(input3, (8, 8))
# sc = MinMaxScaler(feature_range=(0, 16))
# input3 = sc.fit_transform(input3)
# cv2.imwrite('./Adaboost/out.png', input3)

print(model.predict(input2.reshape(1, 64)))
