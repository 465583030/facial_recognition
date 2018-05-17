from face_recognition import get_face
import data_utils as d

pred = get_face(d.get_embeddings([d.load_image('data/test/IMG_1271.JPG', 96)]))
