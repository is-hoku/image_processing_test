import cv2
import numpy as np


def resize_bicubic(img, x, y):
    H, W, C = img.shape
    out = np.zeros((int(y*H), int(x*W), C), dtype=np.float)
    for i in range(int(y*H)):
        for j in range(int(x*W)):
            for k in range(C):
                xi, yi = int(j / x), int(i / y)
                xi = np.minimum(xi, W - 3)
                yi = np.minimum(yi, H - 3)
                dx2 = (j / x) - xi
                dy2 = (i / y) - yi
                dx1 = dx2 + 1
                dy1 = dy2 + 1
                dx3 = 1 - dx2
                dy3 = 1 - dy2
                dx4 = 1 + dx3
                dy4 = 1 + dy3

                def h(t):
                    at = np.abs(t)
                    if at <= 1:
                        t = at ** 3 - 2 * (at ** 2) + 1
                    elif (1 < at) or (at <= 2):
                        t = -at ** 3 + 5 * at ** 2 - 8 * at + 4
                    else:
                        t = 0
                    return t

                if xi < 2:
                    xi = 1
                if yi < 2:
                    yi = 1

                out[i, j, k] = h(dx2)*h(dy2)*img[yi, xi, k] + h(dy2)*h(dx3)*img[yi, xi+1, k] + h(dy3)*h(dx2)*img[yi+1, xi, k] + h(dx3)*h(dx3)*img[yi+1, xi+1, k] + h(dy1)*h(dx2)*img[yi-1, xi, k] + h(dy1)*h(dx3)*img[yi-1, xi+1, k] + h(dy1)*h(dx4)*img[yi-1, xi+2, k] + h(dy2)*h(dx4)*img[yi, xi+2 ,k] + h(dy3)*h(dx4)*img[yi+1, xi+2, k] + h(dy4)*h(dx4)*img[yi+2, xi+2, k] + h(dx3)*h(dy4)*img[yi+2, xi+1, k] + h(dy4)*h(dx2)*img[yi+2, xi, k] + h(dy4)*h(dx1)*img[yi+2, xi-1, k] + h(dy3)*h(dx1)*img[yi+1, xi-1, k] + h(dy2)*h(dx1)*img[yi, xi-1, k] + h(dx1)*h(dy1)*img[yi-1, xi-1, k]
    return out


img = cv2.imread('./img/lenna.jpeg')

out = resize_bicubic(img, 2, 2)

cv2.imwrite("./Bicubic_Interpolation/bicubic.png", out)
