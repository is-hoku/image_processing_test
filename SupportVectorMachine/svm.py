from sklearn.svm import LinearSVC
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
from sklearn.preprocessing import MinMaxScaler, StandardScaler


lfw = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
# print(lfw.images.shape)
# print(lfw.data[0], lfw.target[0], lfw.images[0])

# 訓練データとテストデータに分ける
X_train, X_test, y_train, y_test = train_test_split(
    lfw.data, lfw.target, stratify = lfw.target, random_state=0)

# 標準化
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# クラスの初期化と学習
model = LinearSVC()
model.fit(X_train_std, y_train)

# 訓練データとテストデータのスコア
print('正解率(train):{:.3f}'.format(model.score(X_train_std, y_train)))
print('正解率(test):{:.3f}'.format(model.score(X_test_std, y_test)))

bush = cv2.imread('./img/bush.jpg', 0)
print(lfw.target[100])
print(model.predict(bush.reshape(1, 1850)))
