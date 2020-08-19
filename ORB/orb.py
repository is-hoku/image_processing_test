import numpy as np
import cv2


img = cv2.imread('./img/lenna.jpeg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Initiate ORB detector
orb = cv2.ORB_create()

# find the keypoints and descriptors with ORB
# kp, des = orb.detectAndCompute(gray, None)
kp = orb.detect(img)

out = cv2.drawKeypoints(gray, kp, img)

cv2.imwrite('./ORB/orb.png', out)
