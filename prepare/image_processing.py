from skimage.util import img_as_ubyte
from skimage.color import rgb2gray
from skimage.morphology import square, opening
from skimage.draw import rectangle
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np


# img_as_float
# Convert to 64-bit floating point.
#
# img_as_ubyte
# Convert to 8-bit uint.
#
# img_as_uint
# Convert to 16-bit uint.
#
# img_as_int
# Convert to 16-bit int.


def plot_gray_img(filename):
    '''
    Read the image given the file name, transfer to gray image and plot it
    :param filename: image file name, may have suffix as .png or .jpg
    :return: None
    '''
    color_img = plt.imread(filename)  # shape(None, None, 3), scale 0 ~ 255
    gray_img = rgb2gray(color_img)  # shape(None, None), scale 0 ~ 1
    gray_img = img_as_ubyte(gray_img)  # shape(None, None), scale 0 ~ 255
    plt.figure()
    plt.imshow(gray_img, cmap="gray", vmin=0, vmax=255)


def create_new_plot():
    '''
    Create new plot with matplotlib.pyplot
    :return: None

    Method 1:
    fig, ax = plt.subplots(1)

    Method 2:
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect="equal")

    Method 3:
    plt.figure()
    ax = plt.gca()
    '''
    fig, ax = plt.subplots(1)


def create_mask(image):
    '''
    create image mask
    :param image:
    :return:
    '''
    fig, ax = plt.subplots(1)
    plt.imshow(image)

    img = opening(image > 0, square(101))
    origin = (np.nonzero(img)[0].min(), np.nonzero(img)[1].min())

    fig1, ax1 = plt.subplots(1)
    plt.imshow(img)

    mask = np.zeros_like(img)
    height = Counter(Counter(np.nonzero(img)[1]).values()).most_common()[0][0]
    width = Counter(Counter(np.nonzero(img)[0]).values()).most_common()[0][0]
    rr, cc = rectangle(origin, extent=(height, width), shape=img.shape)
    mask[rr, cc] = 1

    fig2, ax2 = plt.subplots(1)
    plt.imshow(mask)

# import numpy as np
# X = np.zeros([100, 100])
# X[:10, :5] = 1
# X[30: 70, 40:70] = 1
# X[70: 72, 40:65] = 1
# Y = X + np.random.choice([0, 1], size=X.shape, p=[3./4, 1./4])
# Y[Y>1] = 1
# plt.figure(); plt.imshow(Y); plt.savefig("img.png", bbox_inches="tight")
# plt.figure(); plt.imshow(opening(Y, square(10))); plt.savefig("img.png", bbox_inches="tight")
# plt.figure(); plt.imshow(dilation(erosion(Y, square(10)), square(10))); plt.savefig("img.png", bbox_inches="tight")

# img = opening(Y, square(11))
# origin = (np.nonzero(img)[0].min(), np.nonzero(img)[1].min())
# plt.imshow(img)
# plt.scatter(origin[1]-0.5, origin[0]-0.5, color="red")
# plt.savefig("img.png", bbox_inches="tight")
#
# from collections import Counter
#
# height = Counter(Counter(np.nonzero(img)[1]).values()).most_common()[0][0]
# width = Counter(Counter(np.nonzero(img)[0]).values()).most_common()[0][0]
# import matplotlib.patches as patches
#
#
# ax = plt.gca()
# rect = patches.Rectangle((origin[1]-0.5, origin[0]-0.5),width,height,linewidth=1,edgecolor='r',facecolor='none')
#
# from skimage.draw import rectangle
# mask = np.zeros_like(img)
# rr, cc = rectangle(origin, extent=(height, width), shape=img.shape)
# mask[rr, cc] = 1
# plt.imshow(mask, alpha=0.9)
#
# from skimage.draw import rectangle_perimeter
# perimeter = np.zeros_like(img)
# rr, cc = rectangle_perimeter(origin, extent=(height, width), shape=img.shape)
# perimeter[rr, cc] = 1
# plt.imshow(perimeter, alpha=0.1)