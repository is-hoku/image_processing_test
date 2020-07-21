import cv2
# import numpy as np
# import matplotlib.pyplot as plt


img1 = cv2.imread("./img/lenna.jpeg")
img2 = cv2.imread("./img/Parrots.jpg")
img2 = cv2.resize(img2, img1.shape[1::-1])

# dst = cv2.addWeighted(src1, alpha, src2, beta, gamma[, dst[, dtype]])
# dst = src1 * alpha + src2 * beta + gamma
out = cv2.addWeighted(img1, 0.5, img2, 0.5, 0)

cv2.imwrite('./Alpha_Blending/alpha_blend.png', out)
