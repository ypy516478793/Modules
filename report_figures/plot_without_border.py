import matplotlib.pyplot as plt
import numpy as np

def plot_no_border(fileName, image):
    h, w = image.shape[:2]

    fig, ax = plt.subplots(1, figsize=(w/h*5, 1*5))

    # fig = plt.figure(frameon=False)
    # fig.set_size_inches(w/h, 1)

    #
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    #
    # ax.set_ylim(h + 10, -10)
    # ax.set_xlim(-10, w + 10)
    #
    # # ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(image)
    # ax.imshow(image, aspect='auto')
    # fig.savefig(fileName)
    fig.savefig(fileName, dpi=1000)



if __name__ == "__main__":
    A = np.arange(12).reshape(3,4)
    # B = np.random.random([400, 500])
    fileName = "plot_no_boder.png"
    plot_no_border(fileName, A)
    fig2, ax2 = plt.subplots()
    ax2.imshow(A)
    fig2.savefig('out.png', bbox_inches='tight', pad_inches=0)



# fileName = "/home/cougarnet.uh.edu/pyuan2/Downloads/epoch_20.svg"
# img = plt.imread("/home/cougarnet.uh.edu/pyuan2/Ben PC/PycharmProjects/2018 Fall/Seismic/syn_fbk_example/prediction_Trbs5_ep200/epoch_20.jpg")
# plot_no_border(fileName, img)