import cv2
import pickle
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from align_face import align_faces
from model import create_model


def load_raw_data(dir_):
    imgs = []
    for img_dir in dir_:
        img = cv2.imread("test/" + img_dir)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (280, 280))
        imgs.append(img)
    imgs = np.asarray(imgs)
    return get_embeddings(align_faces(imgs))


def load_serial(filename, aligned=False):
    data = pickle.load(open(filename, 'rb'))
    X, y = data['X'], data['y']
    nn = create_model()
    if aligned:
        return nn.predict(align_faces(X))
    else:
        return nn.predict(X)


def get_embeddings(aligned_imgs):
    nn = create_model()
    return nn.predict(aligned_imgs)


data = load_serial(filename='data.pickle')
x = data['x']
y = data['y']
# instantiate knn
knn = KNeighborsClassifier()

knn.fit(X=embs, y=y)

