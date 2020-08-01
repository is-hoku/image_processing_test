import cv2
import numpy as np


def joint_bilateral(flash, non_flash, K_size=3, sigma1=1.3, sigma2=1.3):
    H, W, C = flash.shape
    pad = K_size // 2

    # Zero padding
    out_flash = np.zeros((H + pad * 2, W + pad * 2, C), dtype=np.float)
    out_flash[pad: pad + H, pad: pad + W] = flash.copy().astype(np.float)
    out_non_flash = np.zeros((H + pad * 2, W + pad * 2, C), dtype=np.float)
    out_non_flash[pad: pad + H, pad: pad + W] = non_flash.copy().astype(np.float)
    out = np.zeros((H + pad * 2, W + pad * 2, C), dtype=np.float)

    # kernel
    kernel = np.zeros((K_size, K_size), dtype=np.float)

    # filtering
    for y in range(H):
        for x in range(W):
            for c in range(C):
                for n in range(-pad, -pad + K_size):
                    for m in range(-pad, -pad + K_size):
                        kernel[m + pad, n + pad] = np.exp(-(m ** 2 + n ** 2) / (2 * (sigma1 ** 2))) * np.exp(-((out_flash[y + pad, x + pad, c] - out_flash[y + pad + m, x + pad + n, c]) ** 2) / 2 * (sigma2  ** 2))
                kernel /= np.sum(kernel)
                out[pad + y, pad + x, c] = np.sum(kernel * out_non_flash[y: y + K_size, x: x + K_size, c])

    out = np.clip(out, 0, 255)
    out = out[pad: pad + H, pad: pad + W].astype(np.uint8)

    return out

flash = cv2.imread("./img/lenna.jpeg")
non_flash = cv2.imread("./img/lenna_noize.png")

out = joint_bilateral(flash, non_flash)

cv2.imwrite("./JointBilateral_Filter/joint2.png", out)
