from PIL import Image
import math as m
import numpy as np

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

for x in range(1, len(img_arr) - 1):
    for y in range(1, len(img_arr[x]) - 1):
        reds_grad[x][y] = m.sqrt(m.pow(img_arr[x + 1][y][0] - img_arr[x - 1][y][0], 2) + m.pow(img_arr[x][y + 1][0] - img_arr[x][y - 1][0], 2))
        greens_grad[x][y] = m.sqrt(m.pow(img_arr[x + 1][y][1] - img_arr[x - 1][y][1], 2) + m.pow(img_arr[x][y + 1][1] - img_arr[x][y - 1][1], 2))
        blues_grad[x][y] = m.sqrt(m.pow(img_arr[x + 1][y][2] - img_arr[x - 1][y][2], 2) + m.pow(img_arr[x][y + 1][2] - img_arr[x][y - 1][2], 2))

Image.fromarray(reds_grad).show()
Image.fromarray(greens_grad).show()
Image.fromarray(blues_grad).show()
