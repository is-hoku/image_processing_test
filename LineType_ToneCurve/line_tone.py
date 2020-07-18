import cv2
import numpy as np
# import matplotlib.pyplot as plt


def line_tone1(img, n):
    look_up_table = np.zeros((256, 1), dtype='uint8')
    for i in range(256):
        if i < 256 / n:
            look_up_table[i][0] = i * n
        else:
            look_up_table[i][0] = 255
    return cv2.LUT(img, look_up_table)


def line_tone2(img, n):
    look_up_table = np.zeros((256, 1), dtype='uint8')
    for i in range(256):
        if i < 256 - 256 / n:
            look_up_table[i][0] = 0
        else:
            look_up_table[i][0] = i * n - 255 * (n - 1)
    return cv2.LUT(img, look_up_table)


img = cv2.imread("./img/lenna_gray.jpeg")
out1 = line_tone1(img, 2)
out2 = line_tone2(img, 2)

cv2.imwrite('./4_tone/line_tone1.png', out1)
cv2.imwrite('./4_tone/line_tone2.png', out2)
