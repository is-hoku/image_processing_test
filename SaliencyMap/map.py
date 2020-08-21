import cv2
import numpy as np


def make_pyramid(gray):
    pyramid = [gray]
    H, W = gray.shape

    for i in range(1, 6):
        a = 2. ** i

        p = cv2.resize(gray, (int(W/a), int(H/a)), interpolation=cv2.INTER_LINEAR)
        p = cv2.resize(p, (W, H), interpolation=cv2.INTER_LINEAR)

        pyramid.append(p.astype(np.float32))

    return pyramid


def saliency_map(pyramid):
    H, W = pyramid[0].shape
    out = np.zeros((H, W), dtype=np.float32)

    out += np.abs(pyramid[0] - pyramid[1])
    out += np.abs(pyramid[0] - pyramid[3])
    out += np.abs(pyramid[0] - pyramid[5])
    out += np.abs(pyramid[1] - pyramid[4])
    out += np.abs(pyramid[2] - pyramid[3])
    out += np.abs(pyramid[3] - pyramid[5])

    out = out / out.max() * 255

    return out


img = cv2.imread("./img/glider.jpeg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

pyramid = make_pyramid(gray)

out = saliency_map(pyramid)

out = out.astype(np.uint8)

cv2.imwrite("./SaliencyMap/map.png", out)
