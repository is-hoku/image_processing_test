import cv2
import numpy as np
# import matplotlib.pyplot as plt


def s_tone(img):
    look_up_table = np.zeros((256, 1), dtype='uint8')
    for i in range(256):
        look_up_table[i][0] = 255 * (np.sin(np.pi * (i/255 - 1/2)) + 1) / 2
    return cv2.LUT(img, look_up_table)


img = cv2.imread("./img/lenna_gray.jpeg")
out = s_tone(img)

cv2.imwrite('./S_ToneCurve/s_tone.png', out)
