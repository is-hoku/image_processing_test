import cv2
import numpy as np
# import matplotlib.pyplot as plt


def highpass(img):
    # フーリエ変換
    out = np.fft.fft2(img)
    out = np.fft.fftshift(out)
    # out = 20*np.log(np.abs(out))

    # マスク処理
    img2 = cv2.imread("./img/highmask.png", 0)
    img2 = cv2.resize(img2, out.shape[1::-1])
    out = out*(img2//255)

    # マスク画像を生成からマスク処理までやる場合
    # rows, cols = out.shape
    # crow, ccol = int(rows/2), int(cols/2)
    # fil2 = 20
    # mask = np.zeros((rows, cols), np.uint8)
    # mask[crow-fil2:crow+fil2, ccol-fil2:ccol+fil2] = 0
    # out = out*mask

    # フーリエ逆変換
    out = np.fft.ifftshift(out)
    out = np.fft.ifft2(out)
    out = np.abs(out)
    return out


img = cv2.imread("./img/lenna.jpeg", 0)
out = highpass(img)
cv2.imwrite('./Highpass_Filter/highpass.png', out)
