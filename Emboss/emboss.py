import cv2
import numpy as np
# import matplotlib.pyplot as plt


def emboss(img, x, y):
    # ネガ・ポジ反転
    look_up_table = np.zeros((256, 1), dtype='uint8')
    for i in range(256):
        look_up_table[i][0] = 255 - i
    negaposi = cv2.LUT(img, look_up_table)

    # アフィン変換(平行移動)
    width = img.shape[0]
    height = img.shape[1]
    mat = np.float32([[1, 0, x], [0, 1, y]])
    affine = cv2.warpAffine(negaposi, mat, (height, width))

    # 画像間演算
    out = (affine + img) - 127
    out = out.clip(0, 255)

    return out


img = cv2.imread("./img/lenna_gray.jpeg")
out = emboss(img, 1, 1)

cv2.imwrite('./Emboss/emboss.png', out)
