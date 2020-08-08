import cv2
import numpy as np


def contour(img):
    out = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
    img = cv2.bitwise_not(img)
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, 0, (0, 0, 255), 3)
    return out


img = cv2.imread('./img/a.png')

out = contour(img)

cv2.imwrite("./Contour_Tracking/contour.png", out)
