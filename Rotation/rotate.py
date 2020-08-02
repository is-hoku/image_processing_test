import cv2
import numpy as np


def rotation(img, angle, ratio=1):
    h, w = img.shape[:2]
    src = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]], np.float32)
    dest = np.array([[0.0, 0.0], [np.sin(angle),np.cos(angle)], [np.cos(angle),-np.sin(angle)]], np.float32)
    affine = cv2.getAffineTransform(src, dest)
    # 第1引数は画像，第2引数は変換行列，第3引数は出力画像の大きさ，また補間処理を指定できる．
    out = cv2.warpAffine(img, affine, (w, h), cv2.INTER_LANCZOS4)
    # 領域外の色はborderValue=(Blue, Red, Green)で指定できる
    # out = cv2.warpAffine(img, affine, (w, h), cv2.INTER_LANCZOS4, borderValue=(0, 128, 255))
    # 背景画像はborderModeをcv2.BORDER_TRANSPARENTにして，dstに画像パス
    # out = cv2.warpAffine(img, affine, (w, h), borderMode=cv2.BORDER_TRANSPARENT, dst=img)
    # こういうのとか
    # out = cv2.warpAffine(img, affine, (w, h), borderMode=cv2.BORDER_REPLICATE)
    # こういうのもできる
    # out = cv2.warpAffine(img, affine, (w, h), borderMode=cv2.BORDER_WRAP)

    # 回転のための変換行列を生成する関数
    # 第1引数は回転の原点，第2引数は回転の角度(degree)，第3引数は拡大縮小倍率
    affine = cv2.getRotationMatrix2D((0, 0), angle * 180 / np.pi, ratio)
    out = cv2.warpAffine(img, affine, (w, h), cv2.INTER_LANCZOS4)
    return out


img = cv2.imread('./img/lenna.jpeg')

out1 = rotation(img, np.pi/6)
out2 = rotation(img, np.pi/3)

cv2.imwrite("./Rotation/rotate1.png", out1)
cv2.imwrite("./Rotation/rotate2.png", out2)
