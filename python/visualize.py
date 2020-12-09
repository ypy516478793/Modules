"""
Mask R-CNN
Display and Visualization Functions.

Copyright (c) 2020 HULA, UH, Houston.
Licensed under the MIT License (see LICENSE for details)
Written by Pengyu(Ben) Yuan
"""

from matplotlib import patches, lines
import matplotlib.pyplot as plt
import numpy as np
import colorsys
import random

def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then convert to RGB.
    :param N: number of color to generate
    :param bright: brightness
    :return: None

    example:
        N = 5
        x = np.linspace(0, 2 * np.pi, 50)
        A = np.random.uniform(0, 4, N)
        colors = random_colors(N, bright=False)
        plt.figure()
        for i in range(N):
            line = A[i] * np.sin(x)
            plt.plot(x, line, color=colors[i])
        plt.show()
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors

def canvas2array(fig):
    """
    Save the plot to numpy array
    Run plt.show() or plt.savefig() or fig.canvas.draw() before this function
    :param fig: figure object created by e.g. fig, ax = plt.subplots(1)
    :return: numpy array of the plot
    """
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data

def plot_no_border(image, file_path=None, show=True):
    """
    Plot or save the image without border
    :param image: image, shape == (h, w) or (h, w, c)
    :param file_path: file path for the image to save; normally suffix = ".svg"
    :param show: whether to show in a popup window
    :return: None
    """
    h, w = image.shape[:2]
    fig = plt.figure(figsize=(w / h * 5, 1 * 5))
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(image, aspect='auto')
    if show:
        plt.show()
    else:
        assert isinstance(file_path, str)
        plt.savefig(file_path, dpi=200)
    plt.close(fig)

def draw_bbox(image, bboxes, captions=None, no_border=True):
    """
    Draw the bboxes and return the canvas
    :param image: image, shape == (h, w) or (h, w, c)
    :param bboxes: bounding boxes, shape == (N, 4); dim1 -> (y1, x1, y2, x2)
    :param captions: captions, shape == (N,)
    :param no_border: whether to include border axis in the plot
    :return:
    """
    N = bboxes.shape[0]
    colors = random_colors(N, True)

    ## Matplotlib Axis
    if no_border:
        h, w = image.shape[:2]
        fig, ax = plt.subplots(1, figsize=(w / h * 5, 1 * 5))
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        fig.add_axes(ax)
    else:
        fig, ax = plt.subplots(1)

    ax.imshow(image)
    for i in range(N):
        ## Bounding boxes
        y1, x1, y2, x2 = bboxes[i]
        color = colors[i]
        style = "dotted"  # or "solid"
        alpha = 1
        p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                              alpha=alpha, linestyle=style,
                              edgecolor=color, facecolor='none')
        ax.add_patch(p)

        ## Captions
        if captions is not None:
            caption = captions[i]
            ax.text(x1, y1, caption, size=11, verticalalignment='top',
                    color='w', backgroundcolor="none",
                    bbox={'facecolor': color, 'alpha': 0.5,
                          'pad': 2, 'edgecolor': 'none'})

    fig.canvas.draw()
    return fig

def main():
    A = np.arange(24).reshape(4, 6)
    plot_no_border(A, None, True)
    bboxes = np.array([[0, 1, 2, 4],
                       [1, 3, 3, 4.5]])
    fig = draw_bbox(A, bboxes, no_border=True)
    image = canvas2array(fig)
    plt.close(fig)
    plot_no_border(image, None, True)


def main2():
    A = np.arange(24).reshape(4, 6)
    fig, ax = plt.subplots(1)
    ax.imshow(A, cmap='jet')
    color = random_colors(1, False)[0]
    x1, x2, y1, y2 = 1, 4, 0, 2
    p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                          alpha=0.7, linestyle="dashed",
                          edgecolor=color, facecolor='none')
    ax.add_patch(p)
    fig.canvas.draw()

    data = canvas2array(fig)
    plt.close(fig)
    plt.imshow(data)
    plt.show()


if __name__ == '__main__':
    main2()
    print("")
