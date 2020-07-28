import cv2
import numpy as np
# import matplotlib.pyplot as plt


img = cv2.imread("./PSF_Deterioration/boke2.png", 0)

img = np.fft.fft2(img)
img = np.fft.fftshift(img)
img = 20*np.log(np.abs(img))

fil = [[1/256, 4/256, 6/256, 4/256, 1/256],
       [4/256, 16/256, 24/256, 16/256, 4/256],
       [6/256, 24/256, 36/256, 24/256, 6/256],
       [4/256, 16/256, 24/256, 16/256, 4/256],
       [1/256, 4/256, 6/256, 4/256, 1/256]]

fil = np.fft.fft2(fil)
fil = np.fft.fftshift(fil)
fil = 20*np.log(np.abs(fil))
fil = np.where(fil == 0, 0, 1/fil)

out = cv2.filter2D(img, cv2.CV_64F, fil)

out = np.fft.ifftshift(out)
out = np.fft.ifft2(out)
out = np.abs(out)
print(out)
cv2.imwrite('./PSF_Restoration/boke.png', out)
