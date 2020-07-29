import cv2
import numpy as np
# import matplotlib.pyplot as plt


def CreateImage(img, fil):
    fil = cv2.resize(fil, img.shape[1::-1])

    img = np.fft.fft2(img)
    img = np.fft.fftshift(img)
    # img = 20*np.log(np.abs(img))

    fil = np.fft.fft2(fil)
    fil = np.fft.fftshift(fil)
    fil = 20*np.log(np.abs(fil))
    out = img*(fil/255)

    out = np.fft.ifftshift(out)
    out = np.fft.ifft2(out)
    out = np.abs(out)
    return out


img = cv2.imread("./img/lenna.jpeg", 0)
fil1 = cv2.imread('./img/psf1.png', 0)
fil2 = cv2.imread('./img/psf2.png', 0)
out1 = CreateImage(img, fil1)
out2 = CreateImage(img, fil2)
cv2.imwrite('./PSF_Deterioration/boke.png', out1)
cv2.imwrite('./PSF_Deterioration/bure.png', out2)
