import cv2
import numpy as np
# import matplotlib.pyplot as plt

# フーリエ変換
img = cv2.imread("./img/lenna.jpeg", 0)
out = np.fft.fft2(img)
out = np.fft.fftshift(out)
out = 20*np.log(np.abs(out))

# フーリエ逆変換
# out = np.fft.ifftshift(out)
# out = np.fft.ifft2(out)
# out = np.abs(out)

cv2.imwrite('./Fourier_Transform/fourier.png', out)
