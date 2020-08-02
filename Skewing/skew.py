import cv2
import numpy as np


def skewing(img, shear, dir):
    h, w = img.shape[:2]
    src = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]], np.float32)
    dest = src.copy()
    if dir == 'x':
        dest[:, 0] += (shear / h * (h - src[:, 1])).astype(np.float32)
    elif dir == 'y':
        # dest[:, 1] += (shear / w * (w - src[:, 0])).astype(np.float32)
        dest[:, 1] += (shear / w * src[:, 0]).astype(np.float32)
    affine = cv2.getAffineTransform(src, dest)
    # 第1引数は画像，第2引数は変換行列，第3引数は出力画像の大きさ，また補間処理を指定できる
    out = cv2.warpAffine(img, affine, (w, h), cv2.INTER_LANCZOS4)
    return out


img = cv2.imread('./img/lenna.jpeg')

out1 = skewing(img, 60, 'x')
out2 = skewing(img, 60, 'y')

cv2.imwrite("./Skewing/skew1.png", out1)
cv2.imwrite("./Skewing/skwe2.png", out2)
