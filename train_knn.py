import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import data_utils as d
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from sklearn.manifold import TSNE
from PIL import Image

saved = False
if not saved:
    train_dir = os.listdir('data/train/')
    train_dir.sort()
    if train_dir[0] == '.DS_Store':
        train_dir = train_dir[1:]

    X_train = d.load_raw_images(p_dir='data/train/', dir_=train_dir, save_aligned=False)
    X_train_embs = d.get_embeddings(X_train)
    y_train = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])

    # X_embedded = TSNE(n_components=2).fit_transform(X_train_embs)

    test_dir = os.listdir('data/test/')
    test_dir.sort()
    if test_dir[0] == '.DS_Store':
        test_dir = test_dir[1:]
    X_test = d.load_raw_images(p_dir='data/test/', dir_=test_dir, save_aligned=False)
    X_test_embs = d.get_embeddings(X_test)
    y_test = np.array([2, 3, 2, 2, 2, 3, 3, 3, 2, 2])

    data = {}
    data['X_train_embs'] = X_train_embs
    data['X_train'] = X_train
    data['y_train'] = y_train
    data['X_test_embs'] = X_test_embs
    data['X_test'] = X_test
    data['y_test'] = y_test

    d.save_pickle(data, 'dataset1.pickle')
else:
    data = d.load_serial('dataset1.pickle')
    X_train_embs = data['X_train_embs']
    X_train = data['X_train']
    y_train = data['y_train']
    X_test_embs = data['X_test_embs']
    X_test = data['X_test']
    y_test = data['y_test']

# knn = KNeighborsClassifier(n_neighbors=k)
# svm = LinearSVC()

# knn.fit(X_train_embs, y_train)
# svm.fit(X_train_embs, y_train)

# knn_acc = accuracy_score(y_test, knn.predict(X_test_embs))
# svm_acc = accuracy_score(y_test, svm.predict(X_test_embs))

# print(knn_acc)
# print(svm_acc)

