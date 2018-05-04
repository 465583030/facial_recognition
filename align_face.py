import dlib
import os
import cv2
import numpy as np


def align_test_data(dir_):
    dirs = np.asarray(os.listdir(dir_))
    dirs.sort()
    dirs = dirs[1:]

    imgs = []
    for img_dir in dir_:
        img = cv2.imread("test/" + img_dir)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (280, 280))
        imgs.append(img)

    imgs = np.asarray(imgs)
    a = align_faces(imgs)


def align_faces(images):
    # directory of the landmark data
    dir_landmark = "models/landmarks.dat"

    # retrieve the landmarks from dlib's in order to detect the face
    sp = dlib.shape_predictor(dir_landmark)

    # type of detector
    detector = dlib.get_frontal_face_detector()
    aligned_faces = []

    for image in images:

        dets = detector(image, 1)
        faces = dlib.full_object_detections()
        if len(dets) is 0:
            continue
        else:
            for det in dets:
                faces.append(sp(image, det))
            aligned_face = dlib.get_face_chip(image, faces[0], size=96)

            aligned_face = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
            aligned_faces.append(aligned_face)
    return np.asarray(aligned_faces)
