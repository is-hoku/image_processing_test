import cv2
import numpy as np
# import matplotlib.pyplot as plt


img = cv2.imread("./img/lenna_gray.jpeg")

kernel = np.array([[-1, -1, -1],
                   [-1, 9, -1],
                   [-1, -1, -1]])

out = cv2.filter2D(img, cv2.CV_64F, kernel)

cv2.imwrite('./Sharpning_Filter/sharp.png', out)
