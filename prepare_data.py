import os
import cv2
from PIL import Image
import time
import numpy as np
import pickle
from matplotlib import pyplot as plt

from model import create_model
from align_face import align_faces


def save_imgs(dir_, imgs):
    """
    :param imgs: face images to save
    :param dir_: parent directory of the original images
    """
    t = time.localtime()
    dir_name = dir_ + str(t.tm_mon) + '-' + str(t.tm_mday) + '_' + str(abs(t.tm_hour - 12)) + '-' + str(t.tm_min)
    if not os.path.exists(dir_name):
        print(dir_name)
        os.mkdir(dir_name)

    for i in np.arange(len(imgs)):
        Image.fromarray(imgs[i]).save(dir_name + '/' + '{}.png'.format(i))


def disp_imgs(imgs):
    """
    :param imgs: face images
    """
    f, ax = plt.subplots(5, 5)
    for i in np.arange(5):
        for j in np.arange(5):
            ax[i, j] = imgs[np.random.randint(0, len(imgs))]
    plt.show()


def load_raw_images(p_dir, dir_, save_aligned=False):
    """
    :param p_dir: parent directory
    :param dir_: array or list of image file-names in a targeted directory
    :param save_aligned: save the aligned faces to a directory
    :return: face embeddings of aligned face(s)
    """
    imgs = []
    for img_dir in dir_:
        img = cv2.imread(p_dir + img_dir)
        img = cv2.resize(img, (280, 280))
        imgs.append(img)
    imgs = np.asarray(imgs)
    aligned = align_faces(imgs)
    if save_aligned:
        save_imgs(p_dir, aligned)
    return get_embeddings(aligned)


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
