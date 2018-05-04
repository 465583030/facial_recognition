import os
import cv2
from PIL import Image
import time
import numpy as np
import pickle
from matplotlib import pyplot as plt

from model import create_model
from align_face import align_faces


def save_imgs(imgs):
    """
    :param imgs: face images to save
    """
    t = time.localtime()
    dir_name = os.path.dirname(str(t.tm_mon) + '/' +
                               str(t.tm_mday) + '-' +
                               str(abs(t.tm_hour - 12)) + ':' +
                               str(t.tm_min))
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    for img, i in imgs, np.arange(len(imgs)):
        Image.fromarray(img).save(dir_name + '/' + '{}.png'.format(i))


def disp_imgs(imgs):
    """
    :param imgs: face images
    """
    f, ax = plt.subplots(5, 5)
    for i in np.arange(5):
        for j in np.arange(5):
            ax[i, j] = imgs[np.random.randint(0, len(imgs))]
    plt.show()


def load_raw_images(dir_):
    """
    :param dir_: array or list of image file-names in a targeted directory
    :return: face embeddings of aligned face(s)
    """
    imgs = []
    for img_dir in dir_:
        img = cv2.imread(dir_ + img_dir)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (280, 280))
        imgs.append(img)
    imgs = np.asarray(imgs)
    return get_embeddings(align_faces(imgs))


def load_serial(dir_, aligned=False):
    """
    :param dir_: directory of serialized pickle file being loaded
    :param aligned: whether the faces are aligned or not
    :return: face embeddings of aligned face(s)
    """
    data = pickle.load(open(dir_, 'wb'))
    x, y = data['x'], data['y']
    nn = create_model()
    if aligned:
        return nn.predict(align_faces(x))
    else:
        return nn.predict(x)


def get_embeddings(aligned_imgs):
    """
    :param aligned_imgs: numpy array of aligned images
    :return: face embeddings of aligned face(s)
    """
    nn = create_model()
    return nn.predict(aligned_imgs)
