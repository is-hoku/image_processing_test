import cv2
import numpy as np
# import matplotlib.pyplot as plt


def solarization(img):
    look_up_table = np.zeros((256, 1), dtype='uint8')
    for i in range(256):
        look_up_table[i][0] = (0.5 * (np.sin(10 * (i / 255) - 2 * np.pi)) + 0.5) * 255
    return cv2.LUT(img, look_up_table)


img = cv2.imread("./img/lenna_gray.jpeg")
out = solarization(img)

cv2.imwrite('./Solarization/solarization.png', out)
