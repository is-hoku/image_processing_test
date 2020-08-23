import cv2
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


def dic_color(img):
    # 量子化(32, 96, 160, 224)
    img //= 64
    img = img * 64 + 32
    return img


def get_DB():
    train = glob("./img/train_*")
    train.sort()
    # (y, x) = (テスト画像数, RGB * 4 * class = 13)
    db = np.zeros((len(train), 13), dtype=np.int32)
    pdb = []

    for i, path in enumerate(train):
        img = dic_color(cv2.imread(path))

        for j in range(4):
            db[i, j] = len(np.where(img[..., 0] == (64 * j + 32))[0])
            db[i, j+4] = len(np.where(img[..., 1] == (64 * j + 32))[0])
            db[i, j+8] = len(np.where(img[..., 2] == (64 * j + 32))[0])

        if 'apple' in path:
            cls = 0
        elif 'orange' in path:
            cls = 1

        db[i, -1] = cls
        pdb.append(path)

        # img_h = img.copy() // 64
        # img_h[..., 1] += 4
        # img_h[..., 2] += 8
        # plt.figure()
        # plt.plot(2, 5)
        # plt.hist(img_h.ravel(), bins=12, rwidth=0.8)
        # plt.title(path)
        # plt.savefig("./NearestNeighbor_Method/"+str(i)+".png")

    return db, pdb


def test_DB(db, pdb, N=3):
    test = glob("./img/test_*")
    test.sort()

    accurate_N = 0

    for path in test:
        img = dic_color(cv2.imread(path))

        hist = np.zeros(12, dtype=np.int32)
        for j in range(4):
            hist[j] = len(np.where(img[..., 0] == (64 * j + 32))[0])
            hist[j+4] = len(np.where(img[..., 1] == (64 * j + 32))[0])
            hist[j+8] = len(np.where(img[..., 2] == (64 * j + 32))[0])

        difs = np.abs(db[:, :12] - hist)
        # 行毎の和
        difs = np.sum(difs, axis=1)
        # 最小値のインデックス
        pred_i = np.argsort(difs)[:N]

        pred = db[pred_i, -1]

        if len(pred[pred == 0]) > len(pred[pred == 1]):
            pl = "apple"
        else:
            pl = 'orange'

        print(path, "is similar >> ", end='')
        for i in pred_i:
            print(pdb[i], end=', ')
        print("|Pred >>", pl)

        gt = "apple" if "apple" in path else "orange"

        if gt == pl:
            accurate_N += 1

        accuracy = accurate_N / len(test)
        print("Accuracy >>", accuracy, "({}/{})".format(int(accurate_N), len(test)))


db, pdb = get_DB()
test_DB(db, pdb)
