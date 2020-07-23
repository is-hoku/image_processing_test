import cv2
import numpy as np
# import matplotlib.pyplot as plt


img = cv2.imread("./img/lenna_gray.jpeg")

kernel = np.array([[0, 0, 1, 0, 0],
                   [0, 1, 2, 1, 0],
                   [1, 2, -16, 2, 1],
                   [0, 1, 2, 1, 0],
                   [0, 0, 1, 0, 0]])

out = cv2.filter2D(img, cv2.CV_64F, kernel)

cv2.imwrite('./LoG_Filter/log.png', out)
