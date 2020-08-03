import cv2
import numpy as np


def resize_nearest(img, x, y):
    H, W, C = img.shape
    out = np.zeros((int(y*H), int(x*W), C), dtype=np.float)
    for i in range(int(y*H)):
        for j in range(int(x*W)):
            for k in range(C):
                xi, yi = int(round(j / x)), int(round(i / y))
                if xi >= W:
                    xi = W - 1
                if yi >= H:
                    yi = H - 1
                out[i, j, k] = img[yi, xi, k]
    return out


img = cv2.imread('./img/lenna.jpeg')

out = resize_nearest(img, 2, 2)

cv2.imwrite("./NearestNeighbor/nearest.png", out)
