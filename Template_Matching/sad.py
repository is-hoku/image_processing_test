import cv2
import numpy as np


def SAD(img, tmp):
    H, W, C = img.shape
    Ht, Wt, Ct = tmp.shape

    i, j = -1, -1
    v = 255 ** 2 * Ht * Wt

    for y in range(H - Ht):
        for x in range(W - Wt):
            _v = np.sum(np.abs(img[y: y + Ht, x: x + Wt] - tmp))

            if _v < v:
                v = _v
                i, j = x, y

    out = img.copy()
    cv2.rectangle(out, pt1=(i, j), pt2=(i+Wt, j+Ht), color=(0, 0, 255), thickness=1)
    out = out.astype(np.uint8)

    return out


img = cv2.imread("./img/lenna.jpeg").astype(np.float32)
tmp = cv2.imread("./img/lenna_tmp.jpeg").astype(np.float32)

out = SAD(img, tmp)

cv2.imwrite("./Template_Matching/sad.png", out)
