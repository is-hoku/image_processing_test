import cv2
# import numpy as np
# import matplotlib.pyplot as plt


img = cv2.imread("./img/lenna_gray.jpeg")

# 第2引数：下側閾値，第3引数：上側閾値，第4引数：Sobelフィルタの大きさ
out = cv2.Canny(img, 100, 200, 3)

cv2.imwrite('./Canny_EdgeDetector/canny.png', out)
