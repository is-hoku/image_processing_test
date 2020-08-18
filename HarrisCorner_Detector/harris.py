import cv2
import numpy as np


def harris_corner(img):
    bgr_img = img
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H, W = img.shape
    # Sobelフィルタをかける
    kernelx = np.array([[-1/8, 0, 1/8],
                       [-1/4, 0, 1/4],
                       [-1/8, 0, 1/8]])
    kernely = np.array([[1/8, 1/4, 1/8],
                        [0, 0, 0],
                        [-1/8, -1/4, -1/8]])

    Sx = cv2.filter2D(img, cv2.CV_64F, kernelx)
    Sy = cv2.filter2D(img, cv2.CV_64F, kernely)

    Sxx = Sx ** 2
    Syy = Sy ** 2
    Sxy = Sx * Sy
    # ガウシアンフィルタをかける
    Sxx = cv2.GaussianBlur(Sxx, (3, 3), 3)
    Syy = cv2.GaussianBlur(Syy, (3, 3), 3)
    Sxy = cv2.GaussianBlur(Sxy, (3, 3), 3)
    # コーナー関数
    out = bgr_img
    k = 0.04
    th = 0.1
    R = (Sxx * Syy - Sxy ** 2) - k * ((Sxx + Syy) ** 2)
    out[R >= np.max(R) * th] = [0, 0, 255]
    out = out.astype(np.uint8)

    return out


img = cv2.imread("./img/thorino.jpg")

out = harris_corner(img)

cv2.imwrite("./HarrisCorner_Detector/harris.png", out)
