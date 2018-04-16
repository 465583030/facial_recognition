import numpy as np
from PIL import Image


class HOG:

    """
        img: image to calculate hog for
        img_shape: shape of the image
        gx: gradient of g in the x-direction
        gy: gradient of g in the y-direction
        g: gradient array
        o: gradient direction array
        o_range: degree range of orientation (ex. double angle: 0-360 degrees)
        block_size: size of image window containing cells
        step_size: step size for the window
        cell_size: size of the cells
        num_bins: number of bins per cell
        _normalize: Whether the output array will be normalized or not
        flatten: Whether the output array will be flattened or not
    """

    img = None
    gx = None
    gy = None
    g = None
    o = None
    img_shape = (128, 128)
    o_range = 360
    block_size = (3, 3)  # in cells
    step_size = 5  # in pixels
    cell_size = (10, 10)  # in pixels
    num_bins = 16
    _normalize = True
    flatten = False

    def hog(self, file):
        self.img = self.load_image(filename=file, resize_shape=self.img_shape)
        self.gx, self.gy = self.gradient(self.img)
        self.g = self.grad_magnitude(self.gy, self.gx)
        self.o = (np.arctan2(self.gy, self.gx) * 180 / np.pi) % self.o_range
        histogram = self.calculate_histograms()
        return histogram

    def calculate_histograms(self):
        c_h, c_l = self.cell_size[1], self.cell_size[0]
        b_h, b_l = self.block_size[1] * c_h, self.block_size[0] * c_l
        i_h, i_l = self.g.shape[1], self.g.shape[0]
        histogram = []
        # slide a window across the image to get blocks
        for by in range(0, i_h - (i_h % b_h), b_h):
            for bx in range(0, i_l - (i_l % b_l), b_l):
                block_hist = []
                # slide a window across each block to cells
                for cy in range(by, by + b_h - (b_h % c_h), self.step_size):
                    for cx in range(bx, bx + b_l - (b_l % c_l), self.step_size):
                        gc = self.g[cy:cy + c_h, cx:cx + c_l]
                        oc = self.o[cy:cy + c_h, cx:cx + c_l]
                        gx = self.gx[cy:cy + c_h, cx:cx + c_l]
                        gy = self.gy[cy:cy + c_h, cx:cx + c_l]
                        cell = np.zeros(self.num_bins)
                        for b in np.arange(self.num_bins):
                            t_0 = (b + 1) * 360 / self.num_bins
                            c1 = (oc > (b * t_0 - t_0 / 2)) & (oc < (b * t_0 + t_0 / 2))
                            cell[b] += np.sum(gc[c1]) / gc.size

                            if b > 0:
                                cell[b - 1] += ((np.sin((b + 1) * t_0)/np.sin(t_0)) * np.sum(gx[~c1])) - \
                                               ((np.cos((b + 1) * t_0)/np.sin(t_0)) * np.sum(gy[~c1]))
                            if b < (cell.size - 1):
                                cell[b + 1] += ((np.cos(b * t_0) / np.sin(t_0)) * np.sum(gy[~c1])) - \
                                               ((np.sin(b * t_0) / np.sin(t_0)) * np.sum(gx[~c1]))
                        block_hist.append(cell)

                block_hist = np.asarray(block_hist)

                if self._normalize:
                    block_hist = self.normalize(block_hist)
                if self.flatten:
                    block_hist = block_hist.flatten()

                histogram.append(block_hist)
        histogram = np.asarray(histogram)
        return histogram

    @staticmethod
    def normalize(cell_hist):
        e = 1e-7
        n_hog = np.zeros(cell_hist.shape, dtype=float)
        for cell in range(len(cell_hist)):
            n_hog[cell] = cell_hist[cell] / np.sqrt(np.linalg.norm(cell_hist[cell]) ** 2 + e ** 2)
        return n_hog

    @staticmethod
    def grad_magnitude(y, x):
        """
        :param y: dx
        :param x: dy
        :return: float value of gradient's magnitude
        """
        # calculate magnitude given dy and dx
        return np.sqrt(y ** 2 + x ** 2)

    @staticmethod
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

    @staticmethod
    def load_image(filename, resize_shape=(256, 256)):
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
