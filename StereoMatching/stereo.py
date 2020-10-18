import numpy as np
import cv2


imgL = cv2.imread('./img/tsukuba_l.png', 0)
imgR = cv2.imread('./img/tsukuba_r.png', 0)

# stereo = cv2.createStereoBM(numDisparities=16, blockSize=15)
stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
out = stereo.compute(imgL, imgR)
cv2.imwrite('./StereoMatching/parallax_img.png', out)
