import cv2
import numpy as np


def scaling(img, ratio):
    h, w = img.shape[:2]
    src = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]], np.float32)
    dest = src * ratio
    affine = cv2.getAffineTransform(src, dest)
    # 第1引数は画像，第2引数は変換行列，第3引数は出力画像の大きさ，また補間処理を指定できる
    out = cv2.warpAffine(img, affine, (2*w, 2*h), cv2.INTER_LANCZOS4)
    return out


img = cv2.imread('./img/lenna.jpeg')

out1 = scaling(img, 2)
out2 = scaling(img, 0.5)

cv2.imwrite("./Scaling/expansion.png", out1)
cv2.imwrite("./Scaling/reduction.png", out2)
