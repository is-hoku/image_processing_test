import cv2
import numpy as np
# import matplotlib.pyplot as plt


def posterization(img, n):
    look_up_table = np.zeros((256, 1), dtype='uint8')
    for i in range(256):
        look_up_table[i][0] = (255/(n-1))*(i//(256/n))
    return cv2.LUT(img, look_up_table)


img = cv2.imread("./img/lenna_gray.jpeg")
out = posterization(img, 2)

cv2.imwrite('./Posterization/posterization1.png', out)
