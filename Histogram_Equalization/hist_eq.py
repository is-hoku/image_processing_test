import cv2
# import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread("./img/lenna_gray.jpeg")
gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
out = cv2.equalizeHist(gray_img)

# 処理後のヒストグラムを表示
plt.hist(out.ravel(), bins=255, rwidth=0.8, range=(0, 255))
plt.savefig("./Histogram_Equalization/histogram2.png")
plt.show()

cv2.imwrite('./Histogram_Equalization/hist_eq.png', out)
