import cv2
import numpy as np


img = cv2.imread('./img/paramecium.jpeg')

out = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, out = cv2.threshold(out, 0, 255, cv2.THRESH_OTSU)
kernel = np.ones((5, 5), np.uint8)
out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel)
out = cv2.erode(out, kernel, iterations = 3)
out = cv2.bitwise_not(out)

contours, hierarchy = cv2.findContours(out, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
cnt = contours[0]
M = cv2.moments(out)
# 重心(Red)
cx = int(M['m10']/M['m00'])
cy = int(M['m01']/M['m00'])
img[cy-5: cy+5, cx-5: cx+5, 0] = 0
img[cy-5: cy+5, cx-5: cx+5, 1] = 0
img[cy-5: cy+5, cx-5: cx+5, 2] = 255
# 面積
area = cv2.contourArea(cnt)
# 周辺長
perimeter = cv2.arcLength(cnt, True)
# 外接長方形(Green)
x, y, w, h = cv2.boundingRect(cnt)
cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
# 主軸(Green)
H, W = img.shape[:2]
[vx, vy, x, y] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
lefty = int((-x*vy/vx) + y)
righty = int(((W-x)*vy/vx)+y)
cv2.line(img, (W-1, righty), (0, lefty), (255, 0, 0), 2)

cv2.imwrite("./GeometricFeature_Parameter/moment3.png", img)
