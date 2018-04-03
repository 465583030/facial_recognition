import numpy as np
from PIL import Image
from skimage import io
from matplotlib import pyplot as plt


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
    # calculate dx and dy by splicing img arrays and using [-1, 0, 1] kernel
    x = np.zeros(img.shape)
    y = np.zeros(img.shape)
    x[:, 1:-1] = img[:, 2:] - img[:, :-2]
    y[1:-1, :] = img[2:, :] - img[:-2, :]

    """
    To account for edge cases - however, we want to clip 
    the edges as the gradient cannot be calculated using 
    the 1-D centered filter
    x[:, 0] = -image[:, 0] + image[:, 1]
    x[:, -1] = -image[:, -2] + image[:, -1]
    y[0, :] = -image[0, :] + image[1, :]
    y[-1, :] = -image[-2, :] + image[-1, :]
    """
    # x, y = filters.sobel_h(img), filters.sobel_v(img) # if you want to use 3x3 kernel
    return x, y


def grad_magnitude(x, y):
    """
    :param x: dx
    :param y: dy
    :return: float value of gradient's magnitude
    """
    # calculate magnitude given dx and dy
    return np.sqrt(x**2 + y**2)


def get_windows(img, step_size, window_size):
    w_h = window_size[1]
    w_l = window_size[0]
    i_h = img.shape[1]
    i_l = img.shape[0]

    # slide a window across the image
    for y in range(0, i_h - (i_h % w_h), step_size):
        for x in range(0, i_l - (i_l % w_l), step_size):
            # yield the current window
            window = image[y:y + w_h, x:x + w_l]
            # print("window size error x", window.shape)
            yield (x, y, window)


def calculate_orientation(g, o, step_size, cell_size, num_bins):
    """
    :param g: gradient array
    :param o: gradient direction array
    :param step_size: step size for the window
    :param cell_size: size of the cells
    :param num_bins: number of bins per cell
    :return: 2-D array of histograms per cell
    """
    windows = get_windows(g, step_size, cell_size)
    cells = []
    c_i = 0  # set cell index to 0

    # retrieve generator from get_windows
    for (x, y, window) in windows:
        if window.shape[0] != cell_size[0] or window.shape[1] != cell_size[1]:
            print("window size error x", (window.shape[1], cell_size[1]))
            print("window size error y", (window.shape[0], cell_size[0]))
            continue
        h = np.zeros(num_bins)
        cy, cx = int(y + cell_size[1]), int(x + cell_size[0]/2)
        # t_0 is theta sub naught or the direction of the center pixel of the cell
        t_0 = o[cy, cx]
        for b_i in np.arange(num_bins):
            b = (window > (b_i * t_0 - t_0/2)) & (window < (b_i * t_0 + t_0/2))
            h[b_i] = np.sum(window[b])
        cells.insert(c_i, h)
        c_i += 1  # increment cell index
    return np.asarray(cells)


# def normalization():

image = load_image(filename="test_images/test.jpg", resize_shape=(256, 256))

dx, dy = gradient(image)
grad = grad_magnitude(dx, dy)
orientation = (np.arctan2(dy, dx) * 180/np.pi) % 360

arr = calculate_orientation(g=grad, o=orientation, step_size=5, cell_size=(10, 10), num_bins=16)
arr = arr.flatten()
# for lel in arr:
#     if np.asarray(lel).any() != 0:
#         print(lel)
# show_image(image)
