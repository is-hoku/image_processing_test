import cv2
# import numpy as np
# import matplotlib.pyplot as plt


img = cv2.imread("./img/lenna_noize.jpeg")

# 第2引数はカーネルの大きさ
out = cv2.medianBlur(img, 3)

cv2.imwrite('./Median_Filter/median.png', out)
