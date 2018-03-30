import numpy as np
from PIL import Image
from skimage import filters
from skimage import io

from PIL import Image

def load_image(filename, resize_shape, bw=True):
    """
    :param filename: image takes the form of width, height
    :param resize_shape: shape to resize the image to
    :param bw: whether the photo is black and white or color True = b&w, False = color
    :return: numpy array created by the image

    Takes image file, converts it to B&W, and then re-sizes if applicable.
    """

    img = Image.open(filename).convert('L')
    if resize_shape is not None:
        img = np.array(Image.fromarray(img).resize(resize_shape))

    return img


def show_image(image):
    """
    :param image: takes in numpy array of image to display
    """
    io.imshow(image)
    io.show()
    # Image.fromarray(image).show()


def gradient(image):
    """
    :param image: Image array data
    :param ch: for color arrays
    :param bw: whether the photo is black and white or color True = b&w, False = color
    :return: gradients for x/y
    """
    x = np.zeros(image.shape)
    x[:, 1:-1] = -image[:, :-2] + image[:, 2:]
    x[:, 0] = -image[:, 0] + image[:, 1]
    x[:, -1] = -image[:, -2] + image[:, -1]

    y = np.zeros(image.shape)
    y[1:-1, :] = -image[:-2, :] + image[2:, :]
    y[0, :] = -image[0, :] + image[1, :]
    y[-1, :] = -image[-2, :] + image[-1, :]

"""
def gradient(image, ch):
    # x = np.zeros(image[ch].shape)
    # x[:, 1:-1] = -image[ch][:, :-2] + image[ch][:, 2:]
    # x[:, 0] = -image[ch][:, 0] + image[ch][:, 1]
    # x[:, -1] = -image[ch][:, -2] + image[ch][:, -1]
    #
    # y = np.zeros(image[ch].shape)
    # y[1:-1, :] = -image[ch][:-2, :] + image[ch][2:, :]
    # y[0, :] = -image[ch][0, :] + image[ch][1, :]
    # y[-1, :] = -image[ch][-2, :] + image[ch][-1, :]

    x = np.zeros(len(image[0]))
    y = np.zeros(len(image[0][0]))

    for i in range(len(image[0])):
        for j in range(len(image[0][0])):
            y[i] = x[i] = image[i][j][ch]

    x[:, 1:-1] = -x[:, :-2] + x[:, 2:]
    x[:, 0] = -x[:, 0] + x[:, 1]
    x[:, -1] = -x[:, -2] + x[:, -1]

    y[1:-1, :] = -y[:-2, :] + y[2:, :]
    y[0, :] = -y[0, :] + y[1, :]
    y[-1, :] = -y[-2, :] + y[-1, :]
    return x, y
"""


def grad_magnitude(x, y):
    """
    :param x: gradient in x direction
    :param y: gradient in y direction
    :return: float value of gradient's magnitude
    """
    val = np.sqrt(np.square(x) + np.square(y))
    return val


def max_gradient(image, bw=True):
    """
    :param image: numpy array of a color image
    :param bw: whether the photo is black and white or color True = b&w, False = color
    :return: tuple of gradients

    Gradient of image is computed using the filter [-1, 0, 1] (centered kernel)
    """

    # calculate gradients in both x and y directions
    rx, ry = gradient(image, 0)
    gx, gy = gradient(image, 1)
    bx, by = gradient(image, 2)

    # calculate the magnitude of rgb arrays
    r_mag = grad_magnitude(rx, ry)
    g_mag = grad_magnitude(gx, gy)
    b_mag = grad_magnitude(bx, by)

    grads = [r_mag, g_mag, b_mag]
    norms = [np.linalg.norm(grads[x]) for x in range(len(grads))]

    max_val = grads[norms.index(max(norms))]
    return max_val


def get_thetas(image):

    return image

image = load_image("test_images/test.jpg", resize_shape=(256, 256), bw=True)
max_grad = max_gradient(image, bw=True)
max_grad = np.array(max_grad).astype('uint8')
show_image(max_grad)
