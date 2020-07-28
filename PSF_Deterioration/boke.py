import cv2
# import numpy as np
# import matplotlib.pyplot as plt


img = cv2.imread("./img/lenna.jpeg", 0)

out1 = cv2.GaussianBlur(img, (5, 5), 1)
out2 = cv2.GaussianBlur(img, (5, 5), 6)

cv2.imwrite('./PSF_Deterioration/boke1.png', out1)
cv2.imwrite('./PSF_Deterioration/boke2.png', out2)
