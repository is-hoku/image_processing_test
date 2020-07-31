import cv2
import numpy as np
# import matplotlib.pyplot as plt


def HDR(img1, img2, img3, img4):
    img_list = [img1, img2, img3, img4]
    # exposure_times = np.array([0.0333, 0.25, 2.5, 15.0], dtype=np.float32)
    # Mertensを用いて露光を統合
    merge_mertens = cv2.createMergeMertens()
    res_mertens = merge_mertens.process(img_list)
    out = np.clip(res_mertens*255, 0, 255).astype('uint8')
    return out


img1 = cv2.imread('./img/hdr1.jpeg')
img2 = cv2.imread("./img/hdr2.jpeg")
img3 = cv2.imread('./img/hdr3.jpeg')
img4 = cv2.imread("./img/hdr4.jpeg")

out = HDR(img1, img2, img3, img4)
cv2.imwrite('./HDR_Image/hdr_mertens.png', out)
