import cv2
# import numpy as np
# import matplotlib.pyplot as plt


def DoG(img, ksize, sigma1, sigma2):
    out1 = cv2.GaussianBlur(img, ksize, sigma1)
    out2 = cv2.GaussianBlur(img, ksize, sigma2)
    return out1 - out2


img = cv2.imread("./img/lenna_gray.jpeg")
out = DoG(img, (3, 3), 1, 2)

cv2.imwrite('./DoG_Filter/dog.png', out)
