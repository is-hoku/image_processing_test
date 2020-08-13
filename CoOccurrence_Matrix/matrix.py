import cv2
import numpy as np


def cooccurrence_0(img):
    H, W = img.shape
    matrix = np.zeros((n, n))
    for y in range(H):
        for x in range(W-1):
            matrix[img[y, x], img[y, x+1]] += 1
            matrix[img[y, x+1], img[y, x]] += 1

    return matrix


def cooccurrence_45(img):
    H, W = img.shape
    matrix = np.zeros((n, n))
    for y in range(H-1):
        for x in range(1, W):
            matrix[img[y, x], img[y+1, x-1]] += 1
            matrix[img[y+1, x-1], img[y, x]] += 1

    return matrix


def cooccurrence_90(img):
    H, W = img.shape
    matrix = np.zeros((n, n))
    for y in range(H-1):
        for x in range(W):
            matrix[img[y, x], img[y+1, x]] += 1
            matrix[img[y+1, x], img[y, x]] += 1

    return matrix


def cooccurrence_135(img):
    H, W = img.shape
    matrix = np.zeros((n, n))
    for y in range(H-1):
        for x in range(W-1):
            matrix[img[y, x], img[y+1, x+1]] += 1
            matrix[img[y+1, x+1], img[y, x]] += 1

    return matrix


def posterization(img, n):
    look_up_table = np.zeros((256, 1), dtype='uint8')
    for i in range(256):
        look_up_table[i][0] = (255/(n-1))*(i//(256/n))
    return cv2.LUT(img, look_up_table)


img = cv2.imread("./img/lenna.jpeg", 0)
img = [[0, 0, 1, 1],
       [0, 0, 1, 1],
       [0, 2, 2, 2],
       [2, 2, 3, 3]]
img = np.array(img)

n = 4
out = img
# out = posterization(img, n)
# out = out // (out.max() // (n-1))
out0 = cooccurrence_0(out)
print(out0)
out45 = cooccurrence_45(out)
print(out45)
out90 = cooccurrence_90(out)
print(out90)
out135 = cooccurrence_135(out)
print(out135)
