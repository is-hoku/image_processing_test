import cv2
# import numpy as np
# import matplotlib.pyplot as plt


img = cv2.imread("./img/lenna_gray.jpeg")
out1 = cv2.applyColorMap(img, cv2.COLORMAP_JET)
out2 = cv2.applyColorMap(img, cv2.COLORMAP_AUTUMN)
out3 = cv2.applyColorMap(img, cv2.COLORMAP_BONE)
out4 = cv2.applyColorMap(img, cv2.COLORMAP_COOL)
out5 = cv2.applyColorMap(img, cv2.COLORMAP_HOT)
out6 = cv2.applyColorMap(img, cv2.COLORMAP_HSV)
out7 = cv2.applyColorMap(img, cv2.COLORMAP_OCEAN)
out8 = cv2.applyColorMap(img, cv2.COLORMAP_PINK)
out9 = cv2.applyColorMap(img, cv2.COLORMAP_RAINBOW)
out10 = cv2.applyColorMap(img, cv2.COLORMAP_SPRING)
out11 = cv2.applyColorMap(img, cv2.COLORMAP_SUMMER)
out12 = cv2.applyColorMap(img, cv2.COLORMAP_WINTER)

cv2.imwrite('./PseudoColor/pseudo_jet.png', out1)
cv2.imwrite('./PseudoColor/pseudo_autumn.png', out2)
cv2.imwrite('./PseudoColor/pseudo_bone.png', out3)
cv2.imwrite('./PseudoColor/pseudo_cool.png', out4)
cv2.imwrite('./PseudoColor/pseudo_hot.png', out5)
cv2.imwrite('./PseudoColor/pseudo_hsv.png', out6)
cv2.imwrite('./PseudoColor/pseudo_ocean.png', out7)
cv2.imwrite('./PseudoColor/pseudo_pink.png', out8)
cv2.imwrite('./PseudoColor/pseudo_rainbow.png', out9)
cv2.imwrite('./PseudoColor/pseudo_spring.png', out10)
cv2.imwrite('./PseudoColor/pseudo_summer.png', out11)
cv2.imwrite('./PseudoColor/pseudo_winter.png', out12)
