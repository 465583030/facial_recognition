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
    """
    x[:, 0] = img[:, 1] - img[:, 0]
    x[:, -1] = img[:, -1] - img[:, -2]
    y[0, :] = img[1, :] - img[0, :]
    y[-1, :] = img[-1, :] - img[-2, :]

    # x, y = filters.sobel_h(img), filters.sobel_v(img) # if you want to use 3x3 kernel
    return x, y


def grad_magnitude(x, y):
    """
    :param x: dx
    :param y: dy
    :return: float value of gradient's magnitude
    """
    # calculate magnitude given dx and dy
    return np.sqrt(x ** 2 + y ** 2)


def get_blocks(g, o, step_size, block_size, cell_size, num_bins, _normalize=False, flatten=False):
    c_h, c_l = cell_size[1], cell_size[0]
    b_h, b_l = block_size[1] * c_h, block_size[0] * c_l
    i_h, i_l = g.shape[1], g.shape[0]

    # slide a window across the image
    for y in range(0, i_h - (i_h % b_h), b_h):
        for x in range(0, i_l - (i_l % b_l), b_l):
            # yield the current block
            gb = g[y:y + b_h, x:x + b_l]
            ob = o[y:y + b_h, x:x + b_l]
            block_hists = get_block_hists(block=gb, theta=ob, cell_size=cell_size, step_size=step_size,
                                          num_bins=num_bins, _normalize=_normalize, flatten=flatten)
            yield block_hists


def get_block_hists(block, theta, cell_size, step_size, num_bins, _normalize=False, flatten=False):
    c_h, c_l = cell_size[1], cell_size[0]
    b_h, b_l = block.shape[1], block.shape[0]
    cells = []
    thetas = []

    # slide a window across the image
    for y in range(0, b_h - (b_h % c_h), step_size):
        for x in range(0, b_l - (b_l % c_l), step_size):
            # return list of cells
            cg = block[y:y + c_h, x:x + c_l]
            cells.append(cg)
            co = theta[y:y + c_h, x:x + c_l]
            thetas.append(co)

    # define cell center
    cells = np.asarray(cells)
    thetas = np.asarray(thetas)
    cy, cx = int(cell_size[1] / 2), int(cell_size[0] / 2)

    b_hists = []

    # t_0 is theta sub naught or the direction of the center pixel of the cell
    for x in range(cells.size):
        t_0 = (x + 1) * 360/num_bins
        # t_0 = theta[cy, cx]
        c = np.zeros(num_bins)
        for b in np.arange(num_bins):
            val = (cells[x] > (b * t_0 - t_0 / 2)) & (cells[x] < (b * t_0 + t_0 / 2))
            c[b] += np.sum(cells[x][val]) / cells[x].size
            if b is not 0:
                c[b-1] += np.sin(b+1)*t_0*np.sum()
            if b is not cells.size:
                c[b+1] +=
        b_hists.append(c)

    b_hists = np.asarray(b_hists)
    if _normalize:
        b_hists = normalize(b_hists)
    if flatten:
        b_hists = b_hists.flatten()
    return b_hists


def interpolate_orientation(o):
    i_o = np.zeros(o.shape)
    return i_o


def interpolate(g):
    i_g = np.zeros(g.shape)
    return i_g


def normalize(h):
    e = .001
    n_hog = np.zeros(h.shape, dtype=float)
    for cell in range(len(h)):
        n_hog[cell] = h[cell] / np.sqrt(np.linalg.norm(h[cell]) ** 2 + e ** 2)
    return n_hog


def calculate_histogram(g, o, step_size, block_size, cell_size, num_bins, _normalize=False, flatten=False):
    """
    :param g: gradient array
    :param o: gradient direction array
    :param step_size: step size for the window
    :param block_size: size of image window containing cells
    :param cell_size: size of the cells
    :param num_bins: number of bins per cell
    :return: 2-D array of histograms per cell
    :param flatten: Whether the output array will be flattened or not
    :param _normalize: Whether the output array will be normalized or not
    """

    # retrieve generator from get_windows
    blocks = get_blocks(g, o, step_size, block_size, cell_size, num_bins, _normalize=_normalize, flatten=flatten)
    h = []
    for block in blocks:
        h.append(block)
    h = np.array(h)
    return h


# plot the histogram
def show(cells):
    plt.hist(cells, align='mid')
    plt.plot()
    plt.show()


def histogram(file):
    image = load_image(filename=file, resize_shape=(256, 256))
    dx, dy = gradient(image)
    grad = grad_magnitude(dx, dy)
    orientation = (np.arctan2(dy, dx) * 180 / np.pi) % 360
    hog = calculate_histogram(g=grad, o=orientation, step_size=5, block_size=(3, 3), cell_size=(10, 10),
                              num_bins=16, _normalize=True, flatten=True)
    return hog


hist = histogram("test_images/color_gradient.jpg")
