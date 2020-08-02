import cv2
import numpy as np


def reflection(img, dir):
    h, w = img.shape[:2]
    src = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]], np.float32)
    dest = src.copy()
    if dir == 'x':
        dest[:, 0] = w - src[:, 0]
    elif dir == 'y':
        dest[:, 1] = h - src[:, 1]
    elif dir == 'y-x':
        dest[:, 0] = src[:, 1]
        dest[:, 1] = src[:, 0]
    affine = cv2.getAffineTransform(src, dest)
    # 第1引数は画像，第2引数は変換行列，第3引数は出力画像の大きさ，また補間処理を指定できる
    out = cv2.warpAffine(img, affine, (w, h), cv2.INTER_LANCZOS4)
    if dir == 'y-x':
        out = cv2.warpAffine(img, affine, (w - 15, h + 15), cv2.INTER_LANCZOS4)
    return out


img = cv2.imread('./img/lenna.jpeg')

out1 = reflection(img, 'x')
out2 = reflection(img, 'y')
out3 = reflection(img, 'y-x')

cv2.imwrite("./Reflection/x_axis.png", out1)
cv2.imwrite("./Reflection/y_axis.png", out2)
cv2.imwrite("./Reflection/y-x.png", out3)
