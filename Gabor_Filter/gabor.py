import cv2
import numpy as np


def gabor(img, K_size=111, Sigma=10, Gamma=1.2, Lambda=10, Psi=0, angle=0):
    d = K_size // 2
    gabor = np.zeros((K_size, K_size), dtype=np.float32)

    for y in range(K_size):
        for x in range(K_size):
            px = x - d
            py = y - d

            theta = angle / 180. * np.pi

            _x = np.cos(theta) * px + np.sin(theta) * py
            _y = -np.sin(theta) * px + np.cos(theta) * py

            gabor[y, x] = np.exp(-(_x**2 + Gamma**2 * _y**2) / (2 * Sigma**2)) * np.cos(2*np.pi*_x/Lambda + Psi)
            # gabor[y, x] = np.exp(-(px**2 + py**2)/(2*(Sigma**2))) * np.exp(2*np.pi*Lambda*1j*(px*np.cos(theta) + py*np.sin(theta)))

    # フィルタの正規化
    gabor /= np.sum(np.abs(gabor))

    return gabor


img = cv2.imread("./img/lenna_gray.jpeg", 0)

out1 = gabor(img, K_size=111, Sigma=10, Gamma=1.2, Lambda=10, Psi=0, angle=0)
out1 = gabor(img, K_size=11, Sigma=1.5, Gamma=1.2, Lambda=3, Psi=0, angle=0)
# 0から255に調節
out1 = out1 - np.min(out1)
out1 /= np.max(out1)
out1 *= 255
out1 = out1.astype(np.uint8)
out11 = cv2.filter2D(img, cv2.CV_64F, out1)

out2 = gabor(img, K_size=111, Sigma=10, Gamma=1.2, Lambda=10, Psi=0, angle=45)
out2 = gabor(img, K_size=11, Sigma=1.5, Gamma=1.2, Lambda=3, Psi=0, angle=45)
out2 = out2 - np.min(out2)
out2 /= np.max(out2)
out2 *= 255
out2 = out2.astype(np.uint8)
out21 = cv2.filter2D(img, cv2.CV_64F, out2)

out3 = gabor(img, K_size=111, Sigma=10, Gamma=1.2, Lambda=10, Psi=0, angle=90)
out3 = gabor(img, K_size=11, Sigma=1.5, Gamma=1.2, Lambda=3, Psi=0, angle=90)
out3 = out3 - np.min(out3)
out3 /= np.max(out3)
out3 *= 255
out3 = out3.astype(np.uint8)
out31 = cv2.filter2D(img, cv2.CV_64F, out3)

out4 = gabor(img, K_size=111, Sigma=10, Gamma=1.2, Lambda=10, Psi=0, angle=135)
out4 = gabor(img, K_size=11, Sigma=1.5, Gamma=1.2, Lambda=3, Psi=0, angle=135)
out4 = out4 - np.min(out4)
out4 /= np.max(out4)
out4 *= 255
out4 = out4.astype(np.uint8)
out41 = cv2.filter2D(img, cv2.CV_64F, out4)

out5 = out11 + out21 + out31 + out41

cv2.imwrite('./Gabor_Filter/gabor1.png', out1)
cv2.imwrite('./Gabor_Filter/gabor2.png', out2)
cv2.imwrite('./Gabor_Filter/gabor3.png', out3)
cv2.imwrite('./Gabor_Filter/gabor4.png', out4)

cv2.imwrite('./Gabor_Filter/gabor11.png', out11)
cv2.imwrite('./Gabor_Filter/gabor21.png', out21)
cv2.imwrite('./Gabor_Filter/gabor31.png', out31)
cv2.imwrite('./Gabor_Filter/gabor41.png', out41)

cv2.imwrite('./Gabor_Filter/gabor5.png', out5)
