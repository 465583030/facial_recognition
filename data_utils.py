import os
import cv2
import time
import numpy as np
import pickle
import dlib
from matplotlib import pyplot as plt
from sklearn.externals import joblib
from model import create_model


class Align:
    TEMPLATE = np.float32([
        (0.0792396913815, 0.339223741112), (0.0829219487236, 0.456955367943),
        (0.0967927109165, 0.575648016728), (0.122141515615, 0.691921601066),
        (0.168687863544, 0.800341263616), (0.239789390707, 0.895732504778),
        (0.325662452515, 0.977068762493), (0.422318282013, 1.04329000149),
        (0.531777802068, 1.06080371126), (0.641296298053, 1.03981924107),
        (0.738105872266, 0.972268833998), (0.824444363295, 0.889624082279),
        (0.894792677532, 0.792494155836), (0.939395486253, 0.681546643421),
        (0.96111933829, 0.562238253072), (0.970579841181, 0.441758925744),
        (0.971193274221, 0.322118743967), (0.163846223133, 0.249151738053),
        (0.21780354657, 0.204255863861), (0.291299351124, 0.192367318323),
        (0.367460241458, 0.203582210627), (0.4392945113, 0.233135599851),
        (0.586445962425, 0.228141644834), (0.660152671635, 0.195923841854),
        (0.737466449096, 0.182360984545), (0.813236546239, 0.192828009114),
        (0.8707571886, 0.235293377042), (0.51534533827, 0.31863546193),
        (0.516221448289, 0.396200446263), (0.517118861835, 0.473797687758),
        (0.51816430343, 0.553157797772), (0.433701156035, 0.604054457668),
        (0.475501237769, 0.62076344024), (0.520712933176, 0.634268222208),
        (0.565874114041, 0.618796581487), (0.607054002672, 0.60157671656),
        (0.252418718401, 0.331052263829), (0.298663015648, 0.302646354002),
        (0.355749724218, 0.303020650651), (0.403718978315, 0.33867711083),
        (0.352507175597, 0.349987615384), (0.296791759886, 0.350478978225),
        (0.631326076346, 0.334136672344), (0.679073381078, 0.29645404267),
        (0.73597236153, 0.294721285802), (0.782865376271, 0.321305281656),
        (0.740312274764, 0.341849376713), (0.68499850091, 0.343734332172),
        (0.353167761422, 0.746189164237), (0.414587777921, 0.719053835073),
        (0.477677654595, 0.706835892494), (0.522732900812, 0.717092275768),
        (0.569832064287, 0.705414478982), (0.635195811927, 0.71565572516),
        (0.69951672331, 0.739419187253), (0.639447159575, 0.805236879972),
        (0.576410514055, 0.835436670169), (0.525398405766, 0.841706377792),
        (0.47641545769, 0.837505914975), (0.41379548902, 0.810045601727),
        (0.380084785646, 0.749979603086), (0.477955996282, 0.74513234612),
        (0.523389793327, 0.748924302636), (0.571057789237, 0.74332894691),
        (0.672409137852, 0.744177032192), (0.572539621444, 0.776609286626),
        (0.5240106503, 0.783370783245), (0.477561227414, 0.778476346951)])

    TPL_MIN, TPL_MAX = np.min(TEMPLATE, axis=0), np.max(TEMPLATE, axis=0)
    MINMAX_TEMPLATE = (TEMPLATE - TPL_MIN) / (TPL_MAX - TPL_MIN)

    INNER_EYES_AND_BOTTOM_LIP = [39, 42, 57]

    dir_landmark = "models/landmarks.dat"
    
    @staticmethod
    def get_bounding_box(img):

        assert img is not None

        detector = dlib.get_frontal_face_detector()
        try:
            return detector(img, 1)
        except Exception as e:
            print("Error: {}".format(e))
            return []
    
    def get_largest_bounding_box(self, img):

        assert img is not None

        faces = self.get_bounding_box(img)
        if len(faces) > 0:
            return max(faces, key=lambda rect: rect.width() * rect.height())
        else:
            return None

    def get_landmarks(self, img, bounding_box):

        assert img is not None
        assert bounding_box is not None

        sp = dlib.shape_predictor(self.dir_landmark)
        landmarks = sp(img, bounding_box)
        # landmarks = self.predictor(img, bounding_box)
        return list(map(lambda p: (p.x, p.y), landmarks.parts()))

    def align(self, img, output_size=96, landmark_indices=INNER_EYES_AND_BOTTOM_LIP):

        assert img is not None

        bounding_box = self.get_largest_bounding_box(img=img)
        if bounding_box is None:
            show_image(img)
            return None
        landmarks = np.float32(self.get_landmarks(img=img, bounding_box=bounding_box))
        arr_indices = np.array(landmark_indices)
        temp = cv2.getAffineTransform(landmarks[arr_indices], output_size * self.MINMAX_TEMPLATE[arr_indices])
        return cv2.warpAffine(img, temp, (output_size, output_size))


def save_model(model, filename):
    joblib.dump(model, 'models/{}.pkl'.format(filename))


def load_model(filename):
    assert os.path.exists('models/{}.pkl'.format(filename))
    return joblib.load('models/{}.pkl'.format(filename))


def show_image(img):
    cv2.imshow('', img)


def show_pair(emb1, emb2, img1, img2):
    plt.figure(figsize=(8, 3))
    plt.suptitle('{}'.format(emb_distance(emb1, emb2)))
    plt.subplot(121)
    plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    plt.subplot(122)
    plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    plt.show()


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
        cv2.imwrite(dir_name + '/' + '{}.png'.format(i), imgs[i])


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


def load_image(dir_, aligned_size=96):
    align = Align()
    img = cv2.imread(dir_)
    img = cv2.resize(img, (500, 500))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = align.align(img, output_size=aligned_size)
    if img is None:
        return None
    else:
        return img


def load_raw_images(p_dir, aligned_size=96, save_aligned=False):
    """
    :param p_dir: parent directory
    :param aligned_size: size of output aligned images
    :param save_aligned: save the aligned faces to a directory
    :return: face embeddings of aligned face(s)
    """

    assert p_dir is not None

    dir_ = os.listdir(p_dir)
    dir_.sort()
    if dir_[0] == '.DS_Store':
        dir_ = dir_[1:]

    imgs = np.zeros((len(dir_), 96, 96, 3))
    for i in np.arange(len(dir_)):
        align = Align()
        img = cv2.imread(p_dir + dir_[i])
        img = cv2.resize(img, (500, 500))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = align.align(img, aligned_size)
        if img is None:
            continue
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
    nn.load_weights('weights/nn4.small2.v1.h5')
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
