from collections import deque
from skimage.io import imshow

import matplotlib.pyplot as plt
import numpy as np

def soft_state(imgShape, edge, type='Gaussian', sigma=2, inf=2, sup=200):
    # a = np.arange(imgShape[0]).reshape(-1, 1)
    # b = np.tile(a, (1, imgShape[1]))
    b = np.indices(imgShape)[0]
    k = abs(b - edge)
    canvas = np.zeros(shape=imgShape)

    if type == "Gaussian":
        canvas = gaussian(k, 0, sigma)
    elif type == "Linear":
        canvas[k <= inf] = 1
        canvas[(k <= sup) & (k > inf)] = (k[(k <= sup) & (k > inf)] - sup) / (inf - sup)
    else:
        print('error type')
    return canvas

def pick_edge(segMask):
    edge = deque()
    for j, k in enumerate(np.argmax(np.round(segMask), axis=0)):
        edge.append([j, k])
    edge = np.array(edge)
    return edge

def gaussian(x, mu, sig):
    return np.exp(-(x - mu) ** 2 / (2 * sig ** 2))
    # return 1/(sig * np.sqrt(2 * np.pi)) * np.exp(-(x - mu)**2 / (2 * sig**2))

def seg2bnd(segMasks):
    bndMasks = np.zeros(segMasks.shape)
    for i, segMask in enumerate(segMasks):
        edge = pick_edge(segMask)
        bndMasks[i] = soft_state(segMask.shape, edge=edge[:, 1])
    return bndMasks

if __name__ == '__main__':

    # x_values = np.linspace(-6, 6, 120)
    # plt.plot(x_values, gaussian(x_values, 0, 2))

    folder = "/home/cougarnet.uh.edu/pyuan2/Ben PC/PycharmProjects/2018 Fall/Seismic/syn_fbk_example/result/samepadding_noNoise_200imgspicking/divideMax_99.40/"
    import pickle

    with open(folder + 'ytest.pickle', 'rb') as f:
        Y = pickle.load(f)[..., 0]
    # Y_bnd = seg2bnd(Y)
    image = Y[0]
    edge = pick_edge(image)
    boundaryMask = soft_state(image.shape, type='Gaussian', edge=edge[:, 1])
    plt.figure()
    plt.imshow(image)

    ## --------- Set Matplotlib colorbar size to match graph --------- ##
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    plt.figure()
    ax = plt.gca()
    im = ax.imshow(boundaryMask)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    ## --------- Set Matplotlib colorbar size to match graph --------- ##

    plt.figure()
    imshow(boundaryMask)  ## alternative

    plt.show()
    print('')