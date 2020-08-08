import cv2
import numpy as np


def erosion(img, ksize=3):
    ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
    H, W = img.shape
    out = img.copy()
    d = int((ksize - 1) / 2)
    for y in range(H):
        for x in range(W):
            # 近傍に黒い画素が1つでもあれば、注目画素を黒色に塗り替える
            roi = img[y - d: y + d + 1, x - d: x + d + 1]
            if roi.size - np.count_nonzero(roi) > 0:
                out[y][x] = 0
    return out


def dilation(img, ksize=3):
    ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
    H, W = img.shape
    out = img.copy()
    d = int((ksize - 1) / 2)
    for y in range(0, H):
        for x in range(0, W):
            # 近傍に白い画素が1つでもあれば、注目画素を白色に塗り替える
            roi = img[y - d: y + d + 1, x - d: x + d + 1]
            if np.count_nonzero(roi) > 0:
                out[y][x] = 255
    return out


img = cv2.imread('./img/j.png', 0)

out1 = erosion(img)
out2 = dilation(img)

cv2.imwrite("./Erosion_and_Dilation/erosion.png", out1)
cv2.imwrite("./Erosion_and_Dilation/dilation.png", out2)
