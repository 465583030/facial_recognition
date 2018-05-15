import data_utils as d
import os


def get_metadata(data_dir):
    dir_ = os.listdir(data_dir)
    dir_.sort()
    if dir_[0] == '.DS_Store':
        dir_ = dir_[1:]

    imgs = d.load_raw_images(p_dir=data_dir, dir_=dir_, save_aligned=False)
    embs = d.get_embeddings(imgs)
    return imgs, embs
