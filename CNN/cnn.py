import tensorflow.keras as keras
import cv2
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Input, Activation, add, Add, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from tensorflow.python.keras.utils.vis_utils import model_to_dot
from tensorflow.keras.preprocessing.image import ImageDataGenerator

random_state = 42

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.astype('float32') / 255
y_train = np.eye(10)[y_train.astype('int32').flatten()]

x_test = x_test.astype('float32') / 255
y_test = np.eye(10)[y_test.astype('int32').flatten()]

x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=10000)

model = Sequential()

# filters: フィルター (カーネル) の数，kernel_size: フィルターの大きさ，strides: フィルターを動かす幅，padding: パディング，activation: 活性化関数，use_bias: バイアス項の有無
model.add(Conv2D(6, kernel_size=(5, 5), activation='relu',
                 kernel_initializer='he_normal', input_shape=(32, 32, 3)))  # 32x32x3 -> 28x28x6
# pool_size: プーリングする領域のサイズ，strides: ウィンドウを動かす幅，padding: パディング
# poolingにはMaxPooling2D，AveragePooling2D，GlobalMaxPooling2D，GlobalAveragePooling2Dなどがある
model.add(MaxPooling2D(pool_size=(2, 2)))  # 28x28x6 -> 14x14x6
model.add(Conv2D(16, kernel_size=(5, 5), activation='relu',
                 kernel_initializer='he_normal'))  # 14x14x6 -> 10x10x16
model.add(MaxPooling2D(pool_size=(2, 2)))  # 10x10x16 -> 5x5x16
# 3次元をベクトルに変換
model.add(Flatten())  # 5x5x16 -> 400
model.add(Dense(120, activation='relu',
                kernel_initializer='he_normal'))  # 400 ->120
model.add(Dense(84, activation='relu', kernel_initializer='he_normal'))  # 120 ->84
model.add(Dense(10, activation='softmax'))  # 84 ->10

model.compile(
    loss=keras.losses.categorical_crossentropy,
    optimizer='adam',
    metrics=['accuracy']
)

datagen = ImageDataGenerator(
    width_shift_range=0.2,  # 3.1.1 左右にずらす
    height_shift_range=0.2,  # 3.1.2 上下にずらす
    horizontal_flip=True,  # 3.1.3 左右反転
    # 3.2.1 Global Contrast Normalization (GCN) (Falseに設定しているのでここでは使用していない)
    samplewise_center=False,
    samplewise_std_normalization=False,
    zca_whitening=False)  # 3.2.2 Zero-phase Component Analysis (ZCA) Whitening (Falseに設定しているのでここでは使用していない)
# ZCA whiteningなどデータセットを使って計算をする時はdatagen.fit(x_train)が必要
# datagen.fit(x_train)

model.fit_generator(datagen.flow(x_train, y_train, batch_size=100),
                    steps_per_epoch=x_train.shape[0] // 100, epochs=30, validation_data=(x_valid, y_valid))

score = model.evaluate(x_test, y_test)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

img = cv2.imread('./img/dog.jpeg')
img = cv2.resize(img, (32, 32))

img = model.predict(img.reshape(1, 32, 32, 3)/255.0)

print(np.argmax(img))
