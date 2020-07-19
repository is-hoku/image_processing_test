import cv2
import numpy as np
# import matplotlib.pyplot as plt


def gamma_tone(img, r):
    look_up_table = np.zeros((256, 1), dtype='uint8')
    for i in range(256):
        look_up_table[i][0] = 255 * ((i / 255) ** (1 / r))
    return cv2.LUT(img, look_up_table)


img = cv2.imread("./img/lenna_gray.jpeg")
out = gamma_tone(img, 2)

cv2.imwrite('./Gamma_ToneCurve/gamma_tone.png', out)
