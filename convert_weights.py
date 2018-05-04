from model import create_model
from utils import load_weights

nn4 = create_model()

nn4_weights = load_weights()

for name, w in nn4_weights.items():
    if nn4.get_layer(name) is not None:
        nn4.get_layer(name).set_weights(w)

nn4.save_weights('weights/nn4.small2.v1.h5')
