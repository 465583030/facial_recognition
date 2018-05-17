import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import data_utils as d
# from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt


def get_metadata(data_dir):
    imgs = d.load_raw_images(p_dir=data_dir, save_aligned=False)
    embs = d.get_embeddings(imgs)
    return imgs, embs


def load_pickle(data_dir):
    return d.load_serial(data_dir)


if not os.path.exists('dataset.pickle'):
    X_train_imgs, X_train_embs = get_metadata('data/train/')
    y_train = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 0, 0, 0, 0])
    y_train_names = np.array(['Mazin', 'Mazin', 'Mazin', 'Nithin', 'Nithin', 'Nithin', 'Kunal', 'Kunal', 'Kunal', 'Shreyas', 'Shreyas', 'Shreyas', 'Shreyas'])
    # X_embedded = TSNE(n_components=2).fit_transform(X_train_embs)

    X_test_imgs, X_test_embs = get_metadata('data/test/')
    y_test = np.array([2, 3, 2, 2, 2, 3, 3, 3, 2, 2, 0])
    y_test_names = np.array(['Nithin', 'Kunal', 'Nithin', 'Nithin', 'Nithin', 'Kunal', 'Kunal', 'Kunal', 'Nithin', 'Nithin', 'Shreyas'])
    data = {
        'X_train_embs': X_train_embs,
        'X_train_imgs': X_train_imgs,
        'y_train': y_train,
        'X_test_embs': X_test_embs,
        'X_test_imgs': X_test_imgs,
        'y_test': y_test}

    d.save_pickle(data, 'dataset.pickle')

else:
    data = d.load_serial('dataset.pickle')

    X_train_embs = data['X_train_embs']
    X_train_imgs = data['X_train_imgs']
    y_train = data['y_train']
    y_train_names = np.array(['Mazin', 'Mazin', 'Mazin', 'Nithin', 'Nithin', 'Nithin', 'Kunal', 'Kunal', 'Kunal', 'Shreyas', 'Shreyas', 'Shreyas', 'Shreyas'])

    X_test_embs = data['X_test_embs']
    X_test_imgs = data['X_test_imgs']
    y_test = data['y_test']
    y_test_names = np.array(['Nithin', 'Kunal', 'Nithin', 'Nithin', 'Nithin', 'Kunal', 'Kunal', 'Kunal', 'Nithin', 'Nithin', 'Shreyas'])

X_embedded = TSNE(n_components=2).fit_transform(X_test_embs)

for i, t in enumerate(set(y_test_names)):
    idx = y_test_names == t
    plt.scatter(X_embedded[idx, 0], X_embedded[idx, 1], label=t)

plt.legend(bbox_to_anchor=(1, 1))
plt.show()
# knn = KNeighborsClassifier(n_neighbors=1)
# # scores = cross_val_score(knn, X_train_embs, y_train, cv=3, scoring='accuracy')
# # print(scores.mean())
# knn.fit(X_train_embs, y_train)
# pred = knn.predict(X_test_embs)
# print(accuracy_score(y_pred=pred, y_true=y_test))
# model_name = 'knn_face'
# if not os.path.exists('models/{}.pkl'.format(model_name)):
#     d.save_model(knn, 'knn_face')
