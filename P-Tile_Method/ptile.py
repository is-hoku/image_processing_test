import cv2
import numpy as np
import matplotlib.pyplot as plt


def ptile(img, p):
    size = img.shape[0] * img.shape[1]
    ratio = size * p * 0.01
    hist = plt.hist(img.ravel(), bins=255, rwidth=0.8, range=(0, 255))[0]
    # hist = hist[::-1]
    cnt = 0
    for i in hist:
        cnt += i
        if cnt >= ratio:
            break
    out = np.where(img >= cnt, 255, 0)

    return out


img = cv2.imread('./img/text.png', 0)

out = ptile(img, 10)

cv2.imwrite("./P-Tile_Method/p-tile.png", out)
