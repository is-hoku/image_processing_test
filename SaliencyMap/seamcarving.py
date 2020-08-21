import cv2
import numpy as np


def seam_carving(img, gray):
    H, W = gray.shape
    x = 0
    seam = gray
    row = 255 * H + 1
    for i in range(W):
        cnt = 0
        for y in range(H):
            min_idx = np.argmin([gray[y, max(x-1, 0)], gray[y, x], gray[y, min(x+1, W)]])
            min_idx -= 1
            x = min_idx
            cnt += gray[y, x]
        if cnt < row:
            row = cnt
            row_y = y

    seam = np.delete(img, row_y, 1)
    seam_gray = np.delete(gray, row_y, 1)

    return seam, seam_gray


gray = cv2.imread("./SaliencyMap/map.png", 0)
img = cv2.imread("./img/glider.jpeg")
# gray = cv2.imread("./SaliencyMap/seam_gray.png", 0)
# img = cv2.imread("./SaliencyMap/seam.png")

seam, seam_gray = seam_carving(img, gray)

cv2.imwrite("./SaliencyMap/seam.png", seam)
cv2.imwrite("./SaliencyMap/seam_gray.png", seam_gray)
