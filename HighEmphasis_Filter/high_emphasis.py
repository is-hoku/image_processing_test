import cv2
import numpy as np
# import matplotlib.pyplot as plt


def high_emphasis(img, k):
    # フーリエ変換
    out = np.fft.fft2(img)
    out = np.fft.fftshift(out)
    # out = 20*np.log(np.abs(out))

    # マスク処理
    img2 = cv2.imread("./img/highmask.png", 0)
    img2 = cv2.resize(img2, out.shape[1::-1])
    img3 = cv2.imread("./img/lowmask.png", 0)
    img3 = cv2.resize(img2, out.shape[1::-1])
    # ハイパスフィルタを利用した場合
    out = (1 + k * (img2//255)) * out
    # ローパスフィルタを利用した場合
    # out = (k + 1 - k(img3//255)) * out

    # フーリエ逆変換
    out = np.fft.ifftshift(out)
    out = np.fft.ifft2(out)
    out = np.abs(out)
    return out


img = cv2.imread("./img/lenna.jpeg", 0)
out = high_emphasis(img, 2)
cv2.imwrite('./HighEmphasis_Filter/high_emphasis.png', out)
