import bz2
import os
from urllib.request import urlopen
from model import create_model
from keras.models import Model
from keras.layers import Input
from triplet_generator import triplet_generator
import TripletLossLayer as Tl
import pickle


def download_landmarks(file_path):
    url = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'
    decompressor = bz2.BZ2Decompressor()

    with urlopen(url) as src, open(file_path, 'wb') as dst:
        data = src.read(1024)
        while len(data) > 0:
            dst.write(decompressor.decompress(data))
            data = src.read(1024)


dst_dir = 'models'
dst_file = os.path.join(dst_dir, 'landmarks.dat')
if not os.path.exists(dst_file):
    os.makedirs('models')
    download_landmarks(dst_file)

data_dir = 'aligned_data/'
train_data = pickle.load(open(data_dir + "data.p", "rb"))
x_train, y_train = train_data['x'], train_data['y']
# x_test, y_test =

data_aligned = True

# pretrained model -- much easier than training my own data hahah
nn4_small2 = create_model()
nn4_small2.load_weights('weights/nn4.small2.v1.h5')

# if training data then use this and create data generator - but alas we are not doing so
in_a = Input(shape=(96, 96, 3))
in_p = Input(shape=(96, 96, 3))
in_n = Input(shape=(96, 96, 3))

emb_a = nn4_small2(in_a)
emb_p = nn4_small2(in_p)
emb_n = nn4_small2(in_n)

triplet_loss_layer = Tl.TripletLossLayer(alpha=0.2, name='triplet_loss_layer')([emb_a, emb_p, emb_n])
nn4_small2_train = Model([in_a, in_p, in_n], triplet_loss_layer)

generator = triplet_generator()
nn4_small2_train.compile(loss=None, optimizer='Adam')
nn4_small2_train.fit_generator(generator, epochs=20, steps_per_epoch=100)
