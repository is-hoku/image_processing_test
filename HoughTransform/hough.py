import cv2
import numpy as np


def Hough_transform(edge):
    ## Voting
    def voting(edge):
        H, W = edge.shape
        drho = 1
        dtheta = 1

        # get rho max length
        rho_max = np.ceil(np.sqrt(H ** 2 + W ** 2)).astype(np.int)

        # hough table
        hough = np.zeros((rho_max * 2, 180), dtype=np.int)

        # get index of edge
        ind = np.where(edge == 255)

        ## hough transformation
        for y, x in zip(ind[0], ind[1]):
            for theta in range(0, 180, dtheta):
                # get polar coordinat4s
                t = np.pi / 180 * theta
                rho = int(x * np.cos(t) + y * np.sin(t))

                # vote
                hough[rho + rho_max, theta] += 1

        out = hough.astype(np.uint8)

        return out

    # non maximum suppression
    def non_maximum_suppression(hough):
        rho_max, _ = hough.shape

        for y in range(rho_max):
            for x in range(180):
                # get 8 nearest neighbor
                x1 = max(x-1, 0)
                x2 = min(x+2, 180)
                y1 = max(y-1, 0)
                y2 = min(y+2, rho_max-1)
                if np.max(hough[y1:y2, x1:x2]) == hough[y, x] and hough[y, x] != 0:
                    pass
                else:
                    hough[y, x] = 0

        # for hough visualization
        # get top-10 x index of hough table
        ind_x = np.argsort(hough.ravel())[::-1][:20]
        # get y index
        ind_y = ind_x.copy()
        thetas = ind_x % 180
        rhos = ind_y // 180
        _hough = np.zeros_like(hough, dtype=np.int)
        _hough[rhos, thetas] = 255

        return _hough

    def inverse_hough(hough, img):
        H, W, _ = img.shape
        rho_max, _ = hough.shape

        out = img.copy()

        # get x, y index of hough table
        ind_x = np.argsort(hough.ravel())[::-1][:20]
        ind_y = ind_x.copy()
        thetas = ind_x % 180
        rhos = ind_y // 180 - rho_max / 2

        # each theta and rho
        for theta, rho in zip(thetas, rhos):
            # theta[radian] -> angle[degree]
            t = np.pi / 180. * theta

            # hough -> (x,y)
            for x in range(W):
                if np.sin(t) != 0:
                    y = - (np.cos(t) / np.sin(t)) * x + (rho) / np.sin(t)
                    y = int(y)
                if y >= H or y < 0:
                    continue
                out[y, x] = [0, 0, 255]
            for y in range(H):
                if np.cos(t) != 0:
                    x = - (np.sin(t) / np.cos(t)) * y + (rho) / np.cos(t)
                    x = int(x)
                    if x >= W or x < 0:
                        continue
                    out[y, x] = [0, 0, 255]

        out = out.astype(np.uint8)

        return out

    # voting
    hough = voting(edge)
    # non maximum suppression
    hough = non_maximum_suppression(hough)
    # inverse hough
    out = inverse_hough(hough, img)

    return out


img = cv2.imread("./img/thorino.jpg")

edge = cv2.Canny(img, 100, 200, 3)

out = Hough_transform(edge)

out = out.astype(np.uint8)

cv2.imwrite("./HoughTransform/hough.png", out)
