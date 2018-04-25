import hog4 as h
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier as KNN
import numpy as np

# test = "test/IMG_1138.jpeg"
# histogram = h.HOG().hog(test)
#
# pca = PCA(svd_solver='randomized', n_components=16)
# pca_hist = pca.fit_transform(histogram)
#
# plt.plot(np.cumsum(pca.explained_variance_ratio_))
# plt.xlabel('Number of Components')
# plt.ylabel('Cumulative explained variance')

data = np.load("face_data.npy").item()
names = data['names']
y_train = data['y']
X_train = data['faces']
y_test = np.asarray([1])
X_test = h.HOG().pca_hog("test/test.JPG").flatten()
knn = KNN(n_neighbors=17)
knn.fit(X_train, y_train)
pred = knn.predict(X_test)
print(names[pred[0]])
