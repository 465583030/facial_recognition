import os
import data_utils as d
import numpy as np


def load_model(filename):
    assert os.path.exists('models/{}.pkl'.format(filename))
    return d.load_model(filename)


def get_face(emb):
    clf = load_model('knn_face')
    names = np.array(['Shreyas', 'Mazin', 'Nithin', 'Kunal'])
    emb = emb.reshape(1, -1)
    pred = names[clf.predict(emb)]
    return pred
