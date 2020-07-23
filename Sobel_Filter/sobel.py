import cv2
import numpy as np
# import matplotlib.pyplot as plt


img = cv2.imread("./img/lenna_gray.jpeg")

kernelx = np.array([[-1/8, 0, 1/8],
                   [-1/4, 0, 1/4],
                   [-1/8, 0, 1/8]])
kernely = np.array([[1/8, 1/4, 1/8],
                    [0, 0, 0],
                    [-1/8, -1/4, -1/8]])

out1 = cv2.filter2D(img, cv2.CV_64F, kernelx)
out2 = cv2.filter2D(img, cv2.CV_64F, kernely)

cv2.imwrite('./Sobel_Filter/sobel1.png', out1)
cv2.imwrite('./Sobel_Filter/sobel2.png', out2)
