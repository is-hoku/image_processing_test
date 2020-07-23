import cv2
import numpy as np
# import matplotlib.pyplot as plt


img = cv2.imread("./img/lenna_gray.jpeg")

kernelx = np.array([[-1/6, 0, 1/6],
                    [-1/6, 0, 1/6],
                    [-1/6, 0, 1/6]])
kernely = np.array([[1/6, 1/6, 1/6],
                    [0, 0, 0],
                    [-1/6, -1/6, -1/6]])

out1 = cv2.filter2D(img, cv2.CV_64F, kernelx)
out2 = cv2.filter2D(img, cv2.CV_64F, kernely)

cv2.imwrite('./Prewitt_Filter/prewitt1.png', out1)
cv2.imwrite('./Prewitt_Filter/prewitt2.png', out2)
