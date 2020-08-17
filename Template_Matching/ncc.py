import cv2
import numpy as np


def NCC(img, tmp):
    H, W, C = img.shape
    Ht, Wt, Ct = tmp.shape

    i, j = -1, -1
    v = -1

    for y in range(H - Ht):
        for x in range(W - Wt):
            _v = np.sum(img[y: y + Ht, x: x + Wt] * tmp)
            _v /= (np.sqrt(np.sum(img[y: y + Ht, x: x + Wt] ** 2)) * np.sqrt(np.sum(tmp ** 2)))

            if _v > v:
                v = _v
                i, j = x, y

    out = img.copy()
    cv2.rectangle(out, pt1=(i, j), pt2=(i+Wt, j+Ht), color=(0, 0, 255), thickness=1)
    out = out.astype(np.uint8)

    return out


img = cv2.imread("./img/lenna.jpeg").astype(np.float32)
tmp = cv2.imread("./img/lenna_tmp.jpeg").astype(np.float32)

out = NCC(img, tmp)

cv2.imwrite("./Template_Matching/ncc.png", out)
