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


def frame(img, back, front):
    diff1 = abs(img - back)
    diff1 = otsu(diff1)
    diff2 = abs(img - front)
    diff2 = otsu(diff2)
    kernel = np.ones((3, 3), np.uint8)
    diff1 = cv2.morphologyEx(diff1, cv2.MORPH_CLOSE, kernel)
    diff1 = cv2.morphologyEx(diff1, cv2.MORPH_OPEN, kernel)
    diff2 = cv2.morphologyEx(diff2, cv2.MORPH_CLOSE, kernel)
    diff2 = cv2.morphologyEx(diff2, cv2.MORPH_OPEN, kernel)
    diff1 = np.where(diff1==255, True, False)
    diff2 = np.where(diff2==255, True, False)
    diff = diff1 & diff2
    out = diff * img
    return out


img = cv2.imread('./img/usb_moving.jpg', 0).astype(np.float32)
back = cv2.imread('./img/usb_back.jpg', 0).astype(np.float32)
front = cv2.imread('./img/usb_front.jpg', 0).astype(np.float32)

out = frame(img, back, front)

cv2.imwrite("./FrameSubtraction_Method/moving.png", out)
