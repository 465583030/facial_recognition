import os
import cv2
import pickle
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from align_face import align_faces
from model import create_model
import prepare_data

dirs = os.listdir('data/train/')
dirs.sort()
print(dirs)
dirs = dirs[1:]
X_train = prepare_data.load_raw_images(p_dir='data/train/', dir_=dirs, save_aligned=True)
