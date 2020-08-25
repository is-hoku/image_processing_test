from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


src = fetch_olivetti_faces()

print(src.DESCR)

# 400枚 64x64 グレースケール
X_std = src.images.reshape(400, 64 * 64).T
sc = StandardScaler()
X_std = sc.fit_transform(X_std)

pca = PCA(n_components=5)
pca.fit(X_std)
X_pca = pca.transform(X_std)
print('X_pca shape:{}'.format(X_pca.shape))
print('Explained variance ratio:{}'.format(pca.explained_variance_ratio_))

# (-1 ,1)を(0, 255)に正規化
sc = MinMaxScaler(feature_range=(0, 255))
X_pca = sc.fit_transform(X_pca)

X_pca = X_pca.T.reshape(5, 64, 64)

# for i in range(5):
#     x = np.arange(64 * 64)
#     y = X_pca[i, :, :]
#     print(y)
#     y = y.reshape(64 * 64)
#     print(y.shape)
#     plt.scatter(x=x, y=y)
#     plt.savefig("./PrincipalComponent_Analysis/"+str(i)+".png")

cv2.imwrite('./PrincipalComponent_Analysis/p1.png', X_pca[0, :, :])
cv2.imwrite('./PrincipalComponent_Analysis/p2.png', X_pca[1, :, :])
cv2.imwrite('./PrincipalComponent_Analysis/p3.png', X_pca[2, :, :])
cv2.imwrite('./PrincipalComponent_Analysis/p4.png', X_pca[3, :, :])
cv2.imwrite('./PrincipalComponent_Analysis/p5.png', X_pca[4, :, :])
