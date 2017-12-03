from PIL import Image
import math as m
import numpy as np


def norm(arr):
    norm_val = 0
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            norm_val += m.pow(arr[i][j], 2)
    return m.sqrt(norm_val)

img = Image.open("test_images/IMG_1956.jpg")  # .convert('LA')
img = img.resize((256, 256))

img_arr = np.array(img, "int64")

width, height = img.size

print(width, height)

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
reds_grad_x, reds_grad_y, reds_grad = np.zeros((height, width)), np.zeros((height, width)), np.zeros((height, width))
greens_grad_x, greens_grad_y, greens_grad = np.zeros((height, width)), np.zeros((height, width)), np.zeros((height, width))
blues_grad_x, blues_grad_y, blues_grad = np.zeros((height, width)), np.zeros((height, width)), np.zeros((height, width))

# first m.pow term is the gradient in the x direction, second m.pow term is the gradient in the y direction
# getting magnitude of both in order to compute the magnitude of the gradient
# apply centered linear convolution mask [-1, 0, 1] for each channel
for x in range(1, len(img_arr) - 1):
    for y in range(1, len(img_arr[x]) - 1):
        reds_grad_x[x][y] = img_arr[x + 1][y][0] - img_arr[x - 1][y][0]
        reds_grad_y[x][y] = img_arr[x][y + 1][0] - img_arr[x][y - 1][0]
        reds_grad[x][y] = m.sqrt(m.pow(img_arr[x + 1][y][0] - img_arr[x - 1][y][0], 2)
                                 + m.pow(img_arr[x][y + 1][0] - img_arr[x][y - 1][0], 2))

        greens_grad_x[x][y] = img_arr[x + 1][y][1] - img_arr[x - 1][y][1]
        greens_grad_y[x][y] = img_arr[x][y + 1][1] - img_arr[x][y - 1][1]
        greens_grad[x][y] = m.sqrt(m.pow(img_arr[x + 1][y][1] - img_arr[x - 1][y][1], 2)
                                   + m.pow(img_arr[x][y + 1][1] - img_arr[x][y - 1][1], 2))

        blues_grad_x[x][y] = img_arr[x + 1][y][2] - img_arr[x - 1][y][2]
        blues_grad_y[x][y] = img_arr[x][y + 1][2] - img_arr[x][y - 1][2]
        blues_grad[x][y] = m.sqrt(m.pow(img_arr[x + 1][y][2] - img_arr[x - 1][y][2], 2)
                                  + m.pow(img_arr[x][y + 1][2] - img_arr[x][y - 1][2], 2))

# create parallel lists to get the largest gradient using max function
grads = [[reds_grad_x, reds_grad_y, reds_grad],
         [greens_grad_x, greens_grad_y, greens_grad],
         [blues_grad_x, blues_grad_y, blues_grad]]

norms = [norm(reds_grad), norm(greens_grad), norm(blues_grad)]
max_grad = grads[norms.index(max(norms))]
weights = np.zeros(np.shape(max_grad[0]))

# taking arc-tangent of the sum of magnitudes of each cell to solve for angle then convert to degrees in the
# range 0 < x < 180, if denominator of magnitude is 0, take the arc-tangent of the magnitude - though should check this,
# might want to be careful with assuming the denominator can be set to 1

for x in range(len(max_grad[0])):
    for y in range(len(max_grad[0][0])):
        weights[x][y] = m.degrees(m.atan(max_grad[1][x][y]/max_grad[0][x][y])) % 180 if max_grad[0][x][y] != 0 else \
            m.degrees(m.atan(max_grad[1][x][y])) % 180
        print(weights[x][y])

cell = np.zeros((6, 6))
cells = []
block = np.zeros((3, 3))
blocks = []
