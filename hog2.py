import numpy as np
import cv2
import math as m
from PIL import Image


def load_image(filename, resize_shape, bw=True):
    """
    :param filename: image takes the form of width, height
    :param resize_shape: shape to resize the image to
    :param bw: whether the photo is black and white or color True = b&w, False = color
    :return: numpy array created by the image

    Takes image file, converts it to B&W, and then re-sizes if applicable.
    """

    img = cv2.imread(filename)
    if bw is True:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        b, g, r = cv2.split(img)  # get b,g,r
        img = cv2.merge([r, g, b])
    if resize_shape is not None:
        img = np.array(Image.fromarray(img).resize(resize_shape))

    return img


def show_image(image):
    """
    :param image: takes in numpy array of image to display
    """
    img = Image.fromarray(image)
    img.show()


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

    return x, y


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
    if bw is True:
        x, y = gradient(image)
        return grad_magnitude(x, y)  # return the gradient's magnitude

    else:
        r, g, b = Image.fromarray(image).split()  # split the channels so we can work with each individually
        rx, ry = gradient(np.array(r))
        gx, gy = gradient(np.array(g))
        bx, by = gradient(np.array(b))

        # calculate the magnitude of rgb arrays
        r_mag = grad_magnitude(rx, ry)
        g_mag = grad_magnitude(gx, gy)
        b_mag = grad_magnitude(bx, by)

        grads = [r_mag, g_mag, b_mag]
        norms = [np.linalg.norm(grads[x]) for x in range(len(grads))]
        max_val = grads[norms.index(max(norms))]
        return max_val


image = load_image("test_images/test.jpg", resize_shape=(256, 256), bw=True)
max_grad = max_gradient(image, bw=True)
max_grad = np.array(max_grad).astype('uint8')
show_image(max_grad)
