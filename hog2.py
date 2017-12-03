import numpy as np
from PIL import Image
import math as m


def load_image(filename, resize_shape):
    """
    :param filename: image takes the form of width, height
    :param resize_shape: shape to resize the image to
    :return: numpy array created by the image

    Takes image file, converts it to B&W, and then re-sizes if applicable.
    """

    img = Image.open(filename)
    if resize_shape is not None:
        img = img.resize(resize_shape)
    img = np.array(img)
    return img


def show_image(image):
    """
    :param image: takes in numpy array of image to display
    """
    Image.fromarray(image).show()


def gradient(image, ch):
    x = np.zeros(image[ch].shape)
    x[:, 1:-1] = -image[ch][:, :-2] + image[ch][:, 2:]
    x[:, 0] = -image[ch][:, 0] + image[ch][:, 1]
    x[:, -1] = -image[ch][:, -2] + image[ch][:, -1]

    y = np.zeros(image[ch].shape)
    y[1:-1, :] = -image[ch][:-2, :] + image[ch][2:, :]
    y[0, :] = -image[ch][0, :] + image[ch][1, :]
    y[-1, :] = -image[ch][-2, :] + image[ch][-1, :]

    return x, y


def grad_magnitude(x, y):
    """
    :param x: gradient in x direction
    :param y: gradient in y direction
    :return: float value of gradient's magnitude
    """
    val = np.sqrt(np.square(x) + np.square(y))
    return val


def max_gradient(image):
    """
    :param image: numpy array of a color image
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


image = load_image("test_images/test.jpg", resize_shape=(256, 256))
show_image(image)
max_grad = max_gradient(image)
show_image(max_grad)
