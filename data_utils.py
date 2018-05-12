import os
import cv2
import time
import numpy as np
import pickle
from matplotlib import pyplot as plt
from PIL import Image

from model import create_model
from align_face import Align


def show_image(img):
    Image.fromarray(img.astype('uint8')).show()


def show_pair(emb1, emb2, img1, img2):
    plt.figure(figsize=(8,3))
    plt.suptitle('{}'.format(emb_distance(emb1, emb2)))
    plt.subplot(121)
    plt.imshow(img1)
    plt.subplot(122)
    plt.imshow(img2)


def save_imgs(dir_, imgs):
    """
    :param imgs: face images to save
    :param dir_: parent directory of the original images
    """

    assert dir_ is not None
    assert imgs is not None

    t = time.localtime()
    dir_name = dir_ + str(t.tm_mon) + '-' + str(t.tm_mday) + '_' + str(abs(t.tm_hour - 12)) + '-' + str(t.tm_min)
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    for i in np.arange(len(imgs)):
        Image.fromarray(imgs[i]).save(dir_name + '/' + '{}.png'.format(i))


def disp_imgs(imgs):
    """
    :param imgs: face images
    """

    assert imgs is not None

    f, ax = plt.subplots(5, 5)
    for i in np.arange(5):
        for j in np.arange(5):
            ax[i, j] = imgs[np.random.randint(0, len(imgs))]
    plt.show()


def load_raw_images(p_dir, dir_, aligned_size=96, save_aligned=False):
    """
    :param p_dir: parent directory
    :param dir_: array or list of image file-names in a targeted directory
    :param aligned_size: size of output aligned images
    :param save_aligned: save the aligned faces to a directory
    :return: face embeddings of aligned face(s)
    """

    assert dir_ is not None
    assert p_dir is not None

    imgs = np.zeros((len(dir_), 96, 96, 3))
    for i in np.arange(len(dir_)):
        align = Align()
        img = cv2.imread(p_dir + dir_[i])
        img = cv2.resize(img, (500, 500))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = align.align(img, 96)
        imgs[i] = img

    if save_aligned:
        save_imgs(p_dir, imgs)

    return imgs


def load_serial(dir_):
    """
    :param dir_: directory of serialized pickle file being loaded
    :return: face embeddings of aligned face(s)
    """

    assert dir_ is not None

    return pickle.load(open(dir_, 'rb'))


def get_embeddings(aligned_imgs):
    """
    :param aligned_imgs: numpy array of aligned images
    :return: face embeddings of aligned face(s)
    """

    assert aligned_imgs is not None

    embeddings = np.zeros((len(aligned_imgs), 128))

    nn = create_model()
    for i in np.arange(len(embeddings)):
        img = aligned_imgs[i]
        img = (img / 255.).astype(np.float32)
        embeddings[i] = nn.predict(np.expand_dims(img, axis=0))[0]
    return embeddings


def emb_distance(x1, x2):
    return np.sum(np.square(x1-x2))


def save_pickle(d, filename):
    """
    :param d: dictionary containing numpy arrays
    :param filename: name of output file
    """

    assert d is not None
    assert filename is not None

    pickle.dump(d, open(filename, 'wb'))
