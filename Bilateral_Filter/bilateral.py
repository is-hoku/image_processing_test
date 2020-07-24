import cv2
# import numpy as np
# import matplotlib.pyplot as plt


img = cv2.imread("./img/lenna_gray.jpeg")

# 第2引数はカーネルの大きさ，第3引数は画素値におけるバイラテラルフィルタの分散，第4引数は座標におけるバイラテラルフィルタの分散
out = cv2.bilateralFilter(img, 3, 75, 75)

cv2.imwrite('./Bilateral_Filter/bilateral.png', out)
