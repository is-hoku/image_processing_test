import numpy as np
import cv2
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout


(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 入力画像を行列(28x28)からベクトル(長さ784)に変換し，0-255の整数値のデータの値を0-1の値にスケーリング(前処理)
x_train = x_train.reshape(-1, 784)/255.0
x_test = x_test.reshape(-1, 784)/255.0
# 名義尺度の値をone-hot表現へ変換
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


# モデルの「容器」を作成
model = Sequential()

# 「容器」へ各layer（Dense, Activation）を積み重ねていく（追加した順に配置されるので注意）
# 最初のlayerはinput_shapeを指定して、入力するデータの次元を与える必要がある
# Denseでユニットの追加，Activationは活性化関数
# Denseは全結合層で，前のノードの全ての出力が次のノードの入力になる
# Activationは活性化関数の種類を指定でき，relu,sigmoid,softmax(基本的に出力層で使う)などがある

model.add(Dense(512, input_shape=(784,), activation='relu', kernel_initializer='he_normal'))
model.add(Dense(256, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(128, activation='relu', kernel_initializer='he_normal'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(32, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(10, activation='softmax'))

# 最も単純なモデル定義
# model.add(Dense(units=256, input_shape=(784,)))
# model.add(Activation('relu'))
# model.add(Dense(units=100))
# model.add(Activation('relu'))
# model.add(Dense(units=10))
# model.add(Activation('softmax'))

# Denseで活性化関数を指定することもできる
# model.add(Dense(256, input_shape=(784,), activation='relu'))
# model.add(Dense(100, activation='relu'))
# model.add(Dense(10, activation='softmax'))

# モデルの学習方法について指定しておく
# lossで損失関数の指定(categorical_crossentropy: 交差エントロピー誤差関数)
# optimizerで最適化法を指定(sgd: 確率的勾配降下法)
# metricsでニューラルネットワークの精度の評価方法を指定
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# model.compile(loss='categorical_crossentropy',
#               optimizer='sgd',
#               metrics=['accuracy'])

# batch_size: 学習中のパラメータ更新を1回行うにあたって用いるサンプル数(ミニバッチのサイズ)
# epochs: 学習のエポック数(1つの訓練データを何回繰り返して学習させるか)
# verbose: 学習のログを出力するか(0:しない，1:バーで出力，2:エポックごとに出力)
# validation_split: 検証用に用いるデータの割合
model.fit(x_train, y_train,
          batch_size=1000, epochs=10, verbose=1,
          validation_split=0.1)

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

zero = cv2.imread('./img/0.png', 0)
zero = cv2.resize(zero, (28, 28))
zero = model.predict(zero.reshape(-1, 784)/255.0)
print(np.argmax(zero))

one = cv2.imread('./img/1.png', 0)
one = cv2.resize(one, (28, 28))
one = model.predict(one.reshape(-1, 784)/255.0)
print(np.argmax(one))

two = cv2.imread('./img/2.png', 0)
two = cv2.resize(two, (28, 28))
two = model.predict(two.reshape(-1, 784)/255.0)
print(np.argmax(two))

three = cv2.imread('./img/3.png', 0)
three = cv2.resize(three, (28, 28))
three = model.predict(three.reshape(-1, 784)/255.0)
print(np.argmax(three))

four = cv2.imread('./img/4.png', 0)
four = cv2.resize(four, (28, 28))
four = model.predict(four.reshape(-1, 784)/255.0)
print(np.argmax(four))

five = cv2.imread('./img/5.png', 0)
five = cv2.resize(five, (28, 28))
five = model.predict(five.reshape(-1, 784)/255.0)
print(np.argmax(five))

six = cv2.imread('./img/6.png', 0)
six = cv2.resize(six, (28, 28))
six = model.predict(six.reshape(-1, 784)/255.0)
print(np.argmax(six))

seven = cv2.imread('./img/7.png', 0)
seven = cv2.resize(seven, (28, 28))
cv2.imwrite('./NeuralNetwork/7.png', seven)
seven = model.predict(seven.reshape(-1, 784)/255.0)
print(np.argmax(seven))

eight = cv2.imread('./img/8.png', 0)
eight = cv2.resize(eight, (28, 28))
eight = model.predict(eight.reshape(-1, 784)/255.0)
print(np.argmax(eight))

nine = cv2.imread('./img/9.png', 0)
nine = cv2.resize(nine, (28, 28))
nine = model.predict(nine.reshape(-1, 784)/255.0)
print(np.argmax(nine))
