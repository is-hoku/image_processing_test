import cv2
import numpy as np


def otsu(img):
    max_sigma = 0
    max_t = 0
    H, W = img.shape
    # determine threshold
    for _t in range(1, 256):
        v0 = img[np.where(img < _t)]
        if len(v0) > 0:
            m0 = np.mean(v0)
        else:
            m0 = 0
        w0 = len(v0) / (H * W)
        v1 = img[np.where(img >= _t)]
        if len(v1) > 0:
            m1 = np.mean(v1)
        else:
            m1 = 0
        w1 = len(v1) / (H * W)
        sigma = w0 * w1 * ((m0 - m1) ** 2)
        if sigma > max_sigma:
            max_sigma = sigma
            max_t = _t

    # Binarization
    th = max_t
    img[img < th] = 0
    img[img >= th] = 255
    out = img

    return out


img = cv2.imread('./img/lenna.jpeg', 0)

out = otsu(img)

cv2.imwrite("./DiscriminantAnalysis_Method/otsu.png", out)
