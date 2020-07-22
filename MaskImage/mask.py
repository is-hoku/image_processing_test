import cv2
# import numpy as np
# import matplotlib.pyplot as plt


img1 = cv2.imread("./img/lenna_gray.jpeg")
img2 = cv2.imread("./img/mask_horse.png")
img2 = cv2.resize(img2, img1.shape[1::-1])

# 画素値が0~1に制限されAND(乗算)されるため，マスク画像の白の部分のみ残る．
out = cv2.bitwise_and(img1, img2)

cv2.imwrite('./MaskImage/mask.png', out)
