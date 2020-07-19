import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image
img = cv2.imread("./img/lenna_gray.jpeg").astype(np.float)

# Display histogram
plt.hist(img.ravel(), bins=255, rwidth=0.8, range=(0, 255))
plt.savefig("./GrayHistogram/histogram_1.png")
plt.show()
