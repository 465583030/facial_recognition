import hog as h
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier as KNN
import numpy as np
import os

dirs = os.listdir("test/")
hogs = []
for d in dirs:



# test = "test/IMG_1138.jpeg"
# histogram = h.HOG().hog(test)
#
# pca = PCA(svd_solver='randomized', n_components=16)
# pca_hist = pca.fit_transform(histogram)
#
# plt.plot(np.cumsum(pca.explained_variance_ratio_))
# plt.xlabel('Number of Components')
# plt.ylabel('Cumulative explained variance')

faces = []
