import cv2
import numpy as np
# import matplotlib.pyplot as plt


img = cv2.imread("./img/lenna_gray.jpeg")

kernelx = np.array([[0, 0, 0],
                   [-1/2, 0, 1/2],
                   [0, 0, 0]])
kernely = np.array([[0, 1/2, 0],
                    [0, 0, 0],
                    [0, -1/2, 0]])
kernelxy = np.array([[0, 1/2, 0],
                    [-1/2, 0, 1/2],
                    [0, -1/2, 0]])
out1 = cv2.filter2D(img, cv2.CV_64F, kernelx)
out2 = cv2.filter2D(img, cv2.CV_64F, kernely)
out3 = cv2.filter2D(img, cv2.CV_64F, kernelxy)

cv2.imwrite('./Derivative_Filter/derivative1.png', out1)
cv2.imwrite('./Derivative_Filter/derivative2.png', out2)
cv2.imwrite('./Derivative_Filter/derivative3.png', out3)
