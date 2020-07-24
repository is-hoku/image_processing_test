import cv2
# import numpy as np
# import matplotlib.pyplot as plt


img = cv2.imread("./img/lenna_gray.jpeg")

# 第3引数はフィルタの強さ，大きいほどノイズを消せる(10で十分)，第4引数はカラー画像用のフィルタの強さ(第3引数と同じで良い)，第5引数はテンプレートの大きさ(奇数でなければならなく7が推奨)，第6引数は探索ウィンドウの大きさ(奇数でなければならなく21が推奨)
out = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

cv2.imwrite('./NLmeans_Filter/nlmean.png', out)
