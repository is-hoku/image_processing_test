import cv2
# import numpy as np
# import matplotlib.pyplot as plt


img = cv2.imread("./img/lenna_gray.jpeg")

out = cv2.Laplacian(img, cv2.CV_64F, ksize=1)

cv2.imwrite('./Laplacian_Filter/laplacian.png', out)
