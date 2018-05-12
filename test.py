from align_face import Align
import numpy as np
import os
import cv2
from PIL import Image


def show_image(img):
    Image.fromarray(img).show()


train_dir = os.listdir('data/train/')
train_dir.sort()
print(train_dir)
if train_dir[0] == '.DS_Store':
    train_dir = train_dir[1:]
imgs = np.zeros((len(train_dir), 96, 96, 3))
for i in np.arange(len(train_dir)):
    align = Align()
    img = cv2.imread('data/train/{}'.format(train_dir[i]))
    img = cv2.resize(img, (280, 280))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = align.align(img, 96)
    imgs[i] = img
    show_image(img)