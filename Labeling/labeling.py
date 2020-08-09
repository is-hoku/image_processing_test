import cv2
import numpy as np


# 4連結の場合
def labeling(img):
    H, W, C = img.shape

    label = np.zeros((H, W), dtype=np.int)
    label[img[..., 0] > 0] = 1

    # look up table
    LUT = [0] * (H * W)

    n = 1

    for y in range(H):
        for x in range(W):
            # 黒画素をスキップ
            if label[y, x] == 0:
                continue
            # 上の画素
            c3 = label[max(y-1, 0), x]
            # 左の画素
            c5 = label[y, max(x-1, 0)]

            # ラベルがない場合
            if c3 < 2 and c5 < 2:
                n += 1
                label[y, x] = n
            else:
                _vs = [c3, c5]
                vs = [a for a in _vs if a > 1]
                v = min(vs)
                label[y, x] = v

                minv = v
                for _v in vs:
                    if LUT[_v] != 0:
                        minv = min(minv, LUT[_v])
                for _v in vs:
                    LUT[_v] = minv

    count = 1

    # LUTを適用
    for l in range(2, n+1):
        flag = True
        for i in range(n+1):
            if LUT[i] == l:
                if flag:
                    count += 1
                    flag = False
                LUT[i] = count

    # 色つけ
    COLORS = [[0, 0, 255], [0, 255, 0], [255, 0, 0], [255, 255, 0]]
    out = np.zeros((H, W, C), dtype=np.uint8)

    for i, lut in enumerate(LUT[2:]):
        out[label == (i+2)] = COLORS[lut-2]

    return out


img = cv2.imread("./img/seg.png")

out = labeling(img)

cv2.imwrite("./Labeling/labeling.png", out)
