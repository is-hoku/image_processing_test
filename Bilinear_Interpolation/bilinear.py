import cv2
import numpy as np


def resize_bilinear(img, x, y):
    H, W, C = img.shape
    out = np.zeros((int(y*H), int(x*W), C), dtype=np.float)
    for i in range(int(y*H)):
        for j in range(int(x*W)):
            for k in range(C):
                xi, yi = int(j / x), int(i / y)
                xi = np.minimum(xi, W - 2)
                yi = np.minimum(yi, H - 2)

                dx = (j / x) - xi
                dy = (i / y) - yi
                # 距離が小さいほど重みが大きい
                out[i, j, k] = (1 - dy) * (1 - dx) * img[yi, xi, k] + dy * (1 - dx) * img[yi, xi + 1, k] + (1 - dy) * dx * img[yi + 1, xi, k] + dy * dx * img[yi + 1, xi + 1, k]
    return out


img = cv2.imread('./img/lenna.jpeg')

out = resize_bilinear(img, 2, 2)

cv2.imwrite("./Bilinear_Interpolation/bilinear.png", out)
