import numpy as np
from PIL import Image
from skimage import filters
from skimage import io


def load_image(filename, resize_shape):
    """
    :param filename: image takes the form of width, height
    :param resize_shape: shape to resize the image to
    :return: numpy array created by the image

    Takes image file, converts it to B&W, and then re-sizes if applicable.
    """

    img = Image.open(filename).convert('L')
    if resize_shape is not None:
        img = img.resize(resize_shape)
    img = np.array(img)
    return img


def show_image(img):
    """
    :param img: takes in numpy array of image to display
    """
    io.imshow(img)
    io.show()
    # Image.fromarray(image).show()


def gradient(img):
    """
    :param img: takes in numpy array of image to calculate the gradient of
    :return: tuple containing gradients in both x and y directions
    """
    return filters.sobel_h(img), filters.sobel_v(img)


def grad_magnitude(img):
    """
    :param img: image to find the magnitude of gradients of
    :return: float value of gradient's magnitude
    """

    return filters.sobel(img)


def get_windows(image, step_size, window_size):
    # slide a window across the image
    for y in range(0, image.shape[0], step_size):
        for x in range(0, image.shape[1], step_size):
            # yield the current window
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])


def create_cells(img, step_size, cell_size):
    orientation_bins = np.zeros((img.shape[0]/cell_size[0], img.shape[1]/cell_size[1]))
    for (x, y, window) in get_windows(img, step_size, cell_size):
        if window.shape[0] != cell_size[0] or window.shape[1] != cell_size[1]:
            continue
        #orientation_bins[y, x] =
        return orientation_bins


image = load_image("test_images/test.jpg", resize_shape=(256, 256))
dx, dy = gradient(image)
g = grad_magnitude(image)
t = np.arctan(dy/dx)
create_cells(t, 16, (32, 32))
show_image(image)
