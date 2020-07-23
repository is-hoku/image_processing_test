import cv2
# import numpy as np
# import matplotlib.pyplot as plt


img = cv2.imread("./img/lenna_gray.jpeg")

out1 = cv2.GaussianBlur(img, (3, 3), 4)
out2 = cv2.GaussianBlur(img, (5, 5), 4)

cv2.imwrite('./Gaussian_Filter/gaussian1.png', out1)
cv2.imwrite('./Gaussian_Filter/gaussian2.png', out2)
