from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split, cross_val_score
import numpy as np
import cv2
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


lfw = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
print(lfw.DESCR)
print(lfw.target_names)

X_train, X_test, y_train, y_test = train_test_split(
    lfw.data, lfw.target, random_state=66)

model = RandomForestRegressor(random_state=0)

model.fit(X_train, y_train)
print('正解率(train):{:.3f}'.format(model.score(X_train, y_train)))
print('正解率(test):{:.3f}'.format(model.score(X_test, y_test)))

colin = cv2.imread('./img/colin.jpg', 0)

print(lfw.target_names[int(model.predict(colin.reshape(1, 1850)))])
