import numpy as np
import cv2


img = cv2.imread('./img/lenna.jpeg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Initiate STAR detector
# brief = cv2.DescriptorExtractor_create("BRIEF")
star = cv2.xfeatures2d.StarDetector_create()

# Initiate BRIEF extractor
# brief = cv2.DescriptorExtractor_create("BRIEF")
brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()

# find the keypoints with STAR
kp = star.detect(gray, None)

# compute the descriptors with BRIEF
kp, des = brief.compute(gray, kp)

print(brief.descriptorSize())
print(des.shape)

out = cv2.drawKeypoints(gray, kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imwrite('./BRIEF/brief.png', out)
