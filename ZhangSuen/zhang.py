import cv2
import numpy as np


def thinning(img):
    ret, out = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
    # out = cv2.bitwise_not(img)
    # 背景が1，線が0
    out //= 255
    out = out.astype(np.int)
    H, W = img.shape
    while 1:
        p1 = []
        p2 = []
        # ステップ1
        for y in range(1, H-1):
            for x in range(1, W-1):
                # 条件1: 黒画素である
                if out[y, x] == 0:
                    # 条件2: 8近傍を時計まわりに見て，0から1に変わる回数がちょうど1
                    f1 = 0
                    if (out[y-1, x+1] - out[y-1, x]) == 1:
                        f1 += 1
                    if (out[y, x+1] - out[y-1, x+1]) == 1:
                        f1 += 1
                    if (out[y+1, x+1] - out[y, x+1]) == 1:
                        f1 += 1
                    if (out[y+1, x] - out[y+1, x+1]) == 1:
                        f1 += 1
                    if (out[y+1, x-1] - out[y+1, x]) == 1:
                        f1 += 1
                    if (out[y, x-1] - out[y+1, x-1]) == 1:
                        f1 += 1
                    if (out[y-1, x-1] - out[y, x-1]) == 1:
                        f1 += 1
                    if (out[y-1, x] - out[y-1, x-1]) == 1:
                        f1 += 1

                    if f1 != 1:
                        continue
                    # 条件3: 8近傍の中で1の個数が2以上6以下
                    f2 = np.sum(out[y-1:y+2, x-1:x+2])
                    if f2 < 2 or f2 > 6:
                        continue
                    # 条件4: x2，x4，x6のどれかが1
                    if out[y-1, x] + out[y, x+1] + out[y+1, x] < 1:
                        continue
                    # 条件5: x4，x6，x8のどれかが1
                    if out[y, x+1] + out[y+1, x] + out[y, x-1] < 1:
                        continue
                    p1.append([y, x])
        # 記録したピクセルを全て1に変更する
        for v in p1:
            out[v[0], v[1]] = 1

        # ステップ2
        for y in range(1, H-1):
            for x in range(1, W-1):
                # 条件1: 黒画素である
                if out[y, x] == 0:
                    # 条件2: 8近傍を時計まわりに見て，0から1に変わる回数がちょうど1
                    f1 = 0
                    if (out[y-1, x+1] - out[y-1, x]) == 1:
                        f1 += 1
                    if (out[y, x+1] - out[y-1, x+1]) == 1:
                        f1 += 1
                    if (out[y+1, x+1] - out[y, x+1]) == 1:
                        f1 += 1
                    if (out[y+1, x] - out[y+1, x+1]) == 1:
                        f1 += 1
                    if (out[y+1, x-1] - out[y+1, x]) == 1:
                        f1 += 1
                    if (out[y, x-1] - out[y+1, x-1]) == 1:
                        f1 += 1
                    if (out[y-1, x-1] - out[y, x-1]) == 1:
                        f1 += 1
                    if (out[y-1, x] - out[y-1, x-1]) == 1:
                        f1 += 1

                    if f1 != 1:
                        continue
                    # 条件3: 8近傍の中で1の個数が2以上6以下
                    f2 = np.sum(out[y-1:y+2, x-1:x+2])
                    if f2 < 2 or f2 > 6:
                        continue
                    # 条件4: x2，x4，x8のどれかが1
                    if out[y-1, x] + out[y, x+1] + out[y, x-1] < 1:
                        continue
                    # 条件5: x2，x6，x8のどれかが1
                    if out[y-1, x] + out[y+1, x] + out[y, x-1] < 1:
                        continue
                    p2.append([y, x])
        # 記録したピクセルを全て1に変更する
        for v in p2:
            out[v[0], v[1]] = 1
        # 変更する点が無い場合ループを抜ける
        if len(p1) < 1 and len(p2) < 1:
            break

    out = 1 - out
    out = out.astype(np.uint8) * 255

    return out


img = cv2.imread("./img/a.png", 0)

out = thinning(img)

cv2.imwrite("./ZhangSuen/zhang.png", out)
