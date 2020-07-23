import cv2
# import numpy as np
# import matplotlib.pyplot as plt


img = cv2.imread("./img/lenna_gray.jpeg")

out1 = cv2.blur(img, (3, 3))
out2 = cv2.blur(img, (5, 5))

cv2.imwrite('./Averaging_Filter/average1.png', out1)
cv2.imwrite('./Averaging_Filter/average2.png', out2)
