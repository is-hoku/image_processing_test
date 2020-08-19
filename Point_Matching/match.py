import numpy as np
import cv2


img1 = cv2.imread('./img/lenna.jpeg', 0)
img2 = cv2.imread('./img/lenna_match.jpeg', 0)

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# Apply ratio test
good = []
# kは比率，大きいほど対応点が多くなるが誤検出も増える
k = 0.25
for m, n in matches:
    if m.distance < k*n.distance:
        good.append([m])

# cv2.drawMatchesKnn expects list of lists as matches.
img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)

cv2.imwrite('./Point_Matching/match.png', img3)
