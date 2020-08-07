import cv2
import numpy as np


def mosaicing(img1, img2):
    sift = cv2.xfeatures2d.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for match1, match2 in matches:
        if match1.distance < 0.75*match2.distance:
            good.append([match1])

    sift_matches = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)

    src_pts = np.float32([kp1[m[0].queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m[0].trainIdx].pt for m in good]).reshape(-1, 1, 2)
    h, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC)

    height, width = img1.shape
    dst_img = cv2.warpPerspective(img2, h, (img1.shape[1] + img2.shape[1], img1.shape[0]))

    img_stitched = dst_img.copy()
    img_stitched[:img1.shape[0], :img1.shape[1]] = img1

    return sift_matches, img_stitched


img1 = cv2.imread('./img/toyotakosen1.png', 0)

img2 = cv2.imread('./img/toyotakosen2.png', 0)

out1, out2 = mosaicing(img1, img2)

cv2.imwrite("./ImageMosaicing/mosaicing1.png", out1)
cv2.imwrite("./ImageMosaicing/mosaicing2.png", out2)
