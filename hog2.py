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


def gradient(image, ch=None, bw=True):
    """
    :param image: Image array data
    :param ch: for color arrays
    :param bw: whether the photo is black and white or color True = b&w, False = color
    :return: gradients for x/y
    """
    if ch is not None:
        x = np.zeros(image[0].shape)
        x[:, 1:-1] = -image[:, :-2][ch] + image[:, 2:][ch]
        x[:, 0] = -image[:, 0][ch] + image[:, 1][ch]
        x[:, -1] = -image[:, -2][ch] + image[:, -1][ch]

        y = np.zeros(image[0].shape)
        y[1:-1, :] = -image[:-2, :][ch] + image[2:, :][ch]
        y[0, :] = -image[0, :][ch] + image[1, :][ch]
        y[-1, :] = -image[-2, :][ch] + image[-1, :][ch]
    else:
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
        bw = True
        r, g, b = Image.fromarray(image).split()
        rx, ry = gradient(r, 0, bw)
        gx, gy = gradient(g, 1, bw)
        bx, by = gradient(b, 2, bw)

        # calculate the magnitude of rgb arrays
        r_mag = grad_magnitude(rx, ry)
        g_mag = grad_magnitude(gx, gy)
        b_mag = grad_magnitude(bx, by)
        Image.fromarray(r_mag.astype('uint8')).show()
        grads = [r_mag, g_mag, b_mag]
        norms = [np.linalg.norm(grads[x]) for x in range(len(grads))]

        max_val = grads[norms.index(max(norms))]
        return max_val


image = load_image("test_images/test.jpg", resize_shape=(256, 256), bw=False)
max_grad = max_gradient(image, bw=False)
print(max_grad.shape)
max_grad = np.array(max_grad).astype('uint8')
# show_image(max_grad)