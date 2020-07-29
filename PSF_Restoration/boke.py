import cv2
import numpy as np
# import matplotlib.pyplot as plt


def CreateImage(img, fil):
    # 劣化画像を生成
    # img=原画像，fil=点拡がり関数
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


def RestoreImage(rekka, fil):
    # 劣化画像から原画像を復元(点拡がり関数が既知)
    # rekka=劣化画像，fil=点拡がり関数
    fil = cv2.resize(fil, rekka.shape[1::-1])

    rekka = np.fft.fft2(rekka)
    rekka = np.fft.fftshift(rekka)
    # rekka = 20*np.log(np.abs(rekka))

    fil = np.fft.fft2(fil)
    fil = np.fft.fftshift(fil)
    fil = 20*np.log(np.abs(fil))
    fil = np.where(fil == 0, 0, np.reciprocal(fil))
    out = rekka*(fil*255)

    out = np.fft.ifftshift(out)
    out = np.fft.ifft2(out)
    out = np.abs(out)
    return out


# img = cv2.imread('./img/lenna_gray.jpeg', 0)
img = cv2.imread('./img/Parrots.jpg', 0)
fil = cv2.imread("./img/psf1.png", 0)
rekka = CreateImage(img, fil)
cv2.imwrite('./PSF_Restoration/rekka.png', rekka)

orig_img = RestoreImage(rekka, fil)
cv2.imwrite('./PSF_Restoration/orig_img.png', orig_img)
