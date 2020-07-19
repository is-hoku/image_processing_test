import cv2
import numpy as np
# import matplotlib.pyplot as plt


def negaposi(img):
    look_up_table = np.zeros((256, 1), dtype='uint8')
    for i in range(256):
        look_up_table[i][0] = 255 - i
    return cv2.LUT(img, look_up_table)


img = cv2.imread("./img/lenna_gray.jpeg")
out = negaposi(img)

cv2.imwrite('./NegaPosi_Reversal/negaposi.png', out)
