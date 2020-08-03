import cv2
import numpy as np


def affine(img, ratio, angle, dir1, shear, dir2, transx, transy):
    h, w = img.shape[:2]
    src = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]], np.float32)
    dest = src.copy()
    dest = src * ratio
    dest = np.array([[0.0, 0.0], [np.sin(angle), np.cos(angle)], [np.cos(angle), -np.sin(angle)]], np.float32)
    if dir1 == 'x':
        dest[:, 0] = w - dest[:, 0]
    elif dir1 == 'y':
        dest[:, 1] = h - dest[:, 1]
    elif dir1 == 'y-x':
        dest[:, 0] = dest[:, 1]
        dest[:, 1] = dest[:, 0]
    if dir2 == 'x':
        dest[:, 0] += (shear / h * (h - dest[:, 1])).astype(np.float32)
    elif dir2 == 'y':
        dest[:, 1] += (shear / w * dest[:, 0]).astype(np.float32)
    dest[:, 0] += transx
    dest[:, 1] += transy
    affine = cv2.getAffineTransform(src, dest)
    # 第1引数は画像，第2引数は変換行列，第3引数は出力画像の大きさ，また補間処理を指定できる
    out = cv2.warpAffine(img, affine, (w, h), cv2.INTER_LANCZOS4)
    return out


img = cv2.imread('./img/lenna.jpeg')

out1 = affine(img, 1.5, np.pi/12, 'x', 30, 'y-x', 30, 30)
out2 = affine(img, 0.5, np.pi/6, 'y', 60, 'x', 30, 10)

cv2.imwrite("./Affine_Transformation/affine1.png", out1)
cv2.imwrite("./Affine_Transformation/affine2.png", out2)
