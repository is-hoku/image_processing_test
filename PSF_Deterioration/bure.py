import cv2
import numpy as np
# import matplotlib.pyplot as plt


img = cv2.imread("./img/lenna.jpeg", 0)

kernel1 = np.array([[0, 0, 0, 0, 0],
                    [0, 0, 0, 1/3, 0],
                    [0, 0, 1/3, 0, 0],
                    [0, 1/3, 0, 0, 0],
                    [0, 0, 0, 0, 0]])
kernel2 = np.array([[0, 0, 0, 0, 1/5],
                   [0, 0, 0, 1/5, 0],
                   [0, 0, 1/5, 0, 0],
                   [0, 1/5, 0, 0, 0],
                   [1/5, 0, 0, 0, 0]])

out1 = cv2.filter2D(img, cv2.CV_64F, kernel1)
out2 = cv2.filter2D(img, cv2.CV_64F, kernel2)

cv2.imwrite('./PSF_Deterioration/bure1.png', out1)
cv2.imwrite('./PSF_Deterioration/bure2.png', out2)
