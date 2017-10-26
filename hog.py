from PIL import Image
import math as m
import numpy as np


def norm(arr):
    norm_val = 0
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            norm_val += m.pow(arr[i][j], 2)
    return m.sqrt(norm_val)


img = Image.open("test_images/color_gradient.jpg")  # .convert('LA)
img_arr = np.array(img, "int64")

# FOR B&W IMAGES
"""
print("width: ", len(img_arr))
print("length: ", len(img_arr[0]))

fx, fy, fxy = np.zeros((441, 460)), np.zeros((441, 460)), np.zeros((441, 460))

# apply centered linear convolution mask [-1, 0, 1]
for x in range(1, len(img_arr) - 1):
    for y in range(2, len(img_arr[0]) - 1):
        # print("x: ", x, "y: ", y)
        fx[x][y] = img_arr[x + 1][y][0] - img_arr[x-1][y][0]
        fy[x][y] = img_arr[x][y + 1][0] - img_arr[x][y-1][0]
        fxy[x][y] = m.sqrt(m.pow(fx[x][y], 2) + m.pow(fy[x][y], 2))

Image.fromarray(fxy).show()
"""

# FOR COLOR IMAGES
reds_grad = np.zeros((480, 640))
greens_grad = np.zeros((480, 640))
blues_grad = np.zeros((480, 640))

# first m.pow term is the gradient in the x direction, second m.pow term is the gradient in the y direction
# getting magnitude of both in order to compute the magnitude of the gradient
for x in range(1, len(img_arr) - 1):
    for y in range(1, len(img_arr[x]) - 1):
        reds_grad[x][y] = m.sqrt(m.pow(img_arr[x + 1][y][0] - img_arr[x - 1][y][0], 2)
                                 + m.pow(img_arr[x][y + 1][0] - img_arr[x][y - 1][0], 2))

        greens_grad[x][y] = m.sqrt(m.pow(img_arr[x + 1][y][1] - img_arr[x - 1][y][1], 2)
                                   + m.pow(img_arr[x][y + 1][1] - img_arr[x][y - 1][1], 2))

        blues_grad[x][y] = m.sqrt(m.pow(img_arr[x + 1][y][2] - img_arr[x - 1][y][2], 2)
                                  + m.pow(img_arr[x][y + 1][2] - img_arr[x][y - 1][2], 2))

# create parallel arrays to get the largest gradient using max function
grads = [reds_grad, greens_grad, blues_grad]
norms = [norm(reds_grad), norm(greens_grad), norm(blues_grad)]
max_grad = grads[norms.index(max(norms))]

