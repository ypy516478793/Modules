from matplotlib.patches import Circle
from skimage import io

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import numpy as np
import pickle

def plot_figure_1(x, y, edge, out, clipval=0):

    if clipval != 0:
        maxval = np.percentile(x, 100 - clipval / 2)
        minval = np.percentile(x, clipval / 2)
        x = np.clip(x, minval, maxval)

    pradius = 1
    patches_truth = [Circle((e[0], e[1]), radius=int(pradius), color='red') for e in edge]

    fig = plt.figure(figsize=(7, 4))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.set_xticks([])
    ax2.set_yticks([])
    # ax.imshow(data, cmap=cm.Greys, vmin=-1, vmax=1)
    ax1.imshow(x, cmap=cm.Greys)
    for p in patches_truth:
        ax1.add_patch(p)
    ax2.imshow(y, cmap=cm.Greys, vmin=0, vmax=1)
    # plt.show()
    # fig.set_size_inches([3.375, 3.375])
    fig.savefig(out, format="pdf", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("")

# def plot_figure_2(file1_1, file1_2, out):
#     fig = plt.figure(figsize=(7, 7))
#     ax1 = fig.add_subplot(2, 2, 1)
#     ax2 = fig.add_subplot(2, 2, 2)
#     ax3 = fig.add_subplot(2, 2, 3)
#     ax4 = fig.add_subplot(2, 2, 4)
#     ax1.set_xticks([])
#     ax1.set_yticks([])
#     ax2.set_xticks([])
#     ax2.set_yticks([])
#     ax3.set_xticks([])
#     ax3.set_yticks([])
#     ax4.set_xticks([])
#     ax4.set_yticks([])
#
#     tX, tY, pX, figurePath, idx, Y_label, Y_prediction = loadPickle(folder + file1_1)
#     plot_boundary(tX, tY, pX, Y_label, Y_prediction, ax1, ax3)
#
#     tX, tY, pX, figurePath, idx, Y_label, Y_prediction = loadPickle(folder + file1_2)
#     plot_boundary(tX, tY, pX, Y_label, Y_prediction, ax2, ax4)
#
#     fig.savefig(out, dpi=300, bbox_inches='tight')
#     plt.close(fig)

# def plot_boundary(data, mask, pred, Y_label, Y_prediction, ax1, ax2):
#     # plt.ioff()
#
#     badBias_i = np.mean(abs(Y_label[:, -1] - Y_prediction[:, -1]))
#     badAcc_i = np.sum((pred > 0.5).astype(np.float32) == mask) / (mask.shape[0] * mask.shape[1])
#
#     ## plot overlapping boundaries
#     pradius = 1
#     patches_truth = [Circle((e[0], e[1]), radius=int(pradius), color='red') for e in Y_label]
#     patches_pred = [Circle((e[0], e[1]), radius=int(pradius), color='blue') for e in Y_prediction]
#
#     # manager = plt.get_current_fig_manager()
#     # manager.resize(*manager.window.maxsize())
#     # ax.imshow(data[:2000, :, 0], cmap=cm.Greys, vmin=-1, vmax=1)
#     ax1.imshow(data, cmap=cm.Greys)
#     for p in patches_truth:
#         ax1.add_patch(p)
#     for p in patches_pred:
#         ax1.add_patch(p)
#     # plt.show(fig)
#     if badBias_i: ax1.set_title("avg_bias = {:.2f}".format(badBias_i))
#
#     # manager = plt.get_current_fig_manager()
#     # manager.resize(*manager.window.maxsize())
#     ax2.imshow(pred)
#     if badAcc_i: ax2.set_title("seg_acc = {:.2f}".format(badAcc_i))

def plot_figure_2(file, size, out, name, x_ori=None, clipval=0):
    fig = plt.figure(figsize=size)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    tX, tY, pX, figurePath, idx, Y_label, Y_prediction = loadPickle(folder + file)
    if x_ori is not None:
        tX = x_ori
        tX = tX / np.abs(np.max(tX))
    if clipval != 0:
        maxval = np.percentile(tX, 100 - clipval / 2)
        minval = np.percentile(tX, clipval / 2)
        tX = np.clip(tX, minval, maxval)

    badBias_i = np.mean(abs(Y_label[:, -1] - Y_prediction[:, -1]))
    badAcc_i = np.sum((pX > 0.5).astype(np.float32) == tY) / (tY.shape[0] * tY.shape[1]) * 100

    ## plot overlapping boundaries
    pradius = 1
    patches_truth = [Circle((e[0], e[1]), radius=int(pradius), color='red') for e in Y_label]
    patches_pred = [Circle((e[0], e[1]), radius=int(pradius), color='blue') for e in Y_prediction]

    # manager = plt.get_current_fig_manager()
    # manager.resize(*manager.window.maxsize())
    # ax.imshow(data[:2000, :, 0], cmap=cm.Greys, vmin=-1, vmax=1)
    if clipval != 0:
        ax.imshow(tX, cmap=cm.Greys)
    else:
        ax.imshow(tX, cmap=cm.Greys, vmin=-1, vmax=1)
    for p in patches_truth:
        ax.add_patch(p)
    for p in patches_pred:
        ax.add_patch(p)
    if badBias_i: ax.set_title("avg_bias = {:.2f}".format(badBias_i))
    fig.savefig(out + name + "_pick.svg", dpi=300, bbox_inches='tight')
    plt.close(fig)

    fig = plt.figure(figsize=size)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(pX)
    if badAcc_i: ax.set_title("seg_acc = {:.2f}%".format(badAcc_i))
    fig.savefig(out + name + "_pred.svg", dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_figure_3(fpp, npp, size, out, x_ori, clipval=0):
    x_ori = x_ori[3:747, 3:619, 0]
    fig = plt.figure(figsize=size)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    tX, tY, pX, figurePath, idx, Y_label, Y_prediction = loadPickle(folder + npp)

    if clipval != 0:
        maxval = np.percentile(tX, 100 - clipval / 2)
        minval = np.percentile(tX, clipval / 2)
        tX = np.clip(tX, minval, maxval)

    badBias_i = np.mean(abs(Y_label[:, -1] - Y_prediction[:, -1]))
    badAcc_i = np.sum((pX > 0.5).astype(np.float32) == tY) / (tY.shape[0] * tY.shape[1]) * 100

    ## plot overlapping boundaries
    pradius = 1
    patches_truth = [Circle((e[0], e[1]), radius=int(pradius), color='red') for e in Y_label]
    patches_pred = [Circle((e[0], e[1]), radius=int(pradius), color='blue') for e in Y_prediction]
    ax.imshow(tX, cmap=cm.Greys)
    for p in patches_truth:
        ax.add_patch(p)
    for p in patches_pred:
        ax.add_patch(p)
    if badBias_i: ax.set_title("avg_bias = {:.2f}".format(badBias_i))
    fig.savefig(out + "npp" + "_pick.svg", dpi=300, bbox_inches='tight')
    plt.close(fig)

    fig = plt.figure(figsize=size)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(pX)
    if badAcc_i: ax.set_title("seg_acc = {:.2f}%".format(badAcc_i))
    fig.savefig(out + "npp" + "_pred.svg", dpi=300, bbox_inches='tight')
    plt.close(fig)

    fig = plt.figure(figsize=size)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    tX, tY, pX, figurePath, idx, Y_label, Y_prediction = loadPickle(folder + fpp)

    if clipval != 0:
        maxval = np.percentile(tX, 100 - clipval / 2)
        minval = np.percentile(tX, clipval / 2)
        tX = np.clip(tX, minval, maxval)

    badBias_i = np.mean(abs(Y_label[:, -1] - Y_prediction[:, -1]))
    ## plot overlapping boundaries
    pradius = 1
    patches_truth = [Circle((e[0], e[1]), radius=int(pradius), color='red') for e in Y_label]
    patches_pred = [Circle((e[0], e[1]), radius=int(pradius), color='blue') for e in Y_prediction]
    ax.imshow(tX, cmap=cm.Greys)
    for p in patches_truth:
        ax.add_patch(p)
    for p in patches_pred:
        ax.add_patch(p)
    if badBias_i: ax.set_title("avg_bias = {:.2f}".format(badBias_i))
    fig.savefig(out + "fpp" + "_pick.svg", dpi=300, bbox_inches='tight')
    plt.close(fig)

    fig = plt.figure(figsize=size)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(1-tY, cmap=cm.Greys)
    if badAcc_i: ax.set_title("seg_acc = {:.2f}%".format(badAcc_i))
    fig.savefig(out + "gtmask.svg", dpi=300, bbox_inches='tight')
    plt.close(fig)

def savePickle(name, data):
    with open(name, 'wb') as f:
        pickle.dump(data, f)

def loadPickle(name):
    with open(name, 'rb') as f:
        data = pickle.load(f)
    return data

def readTest(fileName):
    import h5py
    with h5py.File(fileName, "r") as f:
        X = f['X_data']
        Y = f['Y_data']
        x, y, z = X.shape
        X = X[:, 0:y//2*2, 0:z//2*2]
        Y = Y[:, 0:y//2*2, 0:z//2*2]
        Y = np.stack([Y, 1-Y], axis=-1)
        X = X[..., np.newaxis]
    return X, Y

def readField(fileName):
    import h5py
    with h5py.File(fileName, "r") as f:
        X, Y = [], []
        X_keys = [key for key in list(f.keys()) if key[0] == 'X']
        X_keys.sort(key=lambda x: int(x[6:]))
        Y_keys = [key for key in list(f.keys()) if key[0] == 'Y']
        Y_keys.sort(key=lambda x: int(x[6:]))


        # [X_keys.remove(i) for i in ["X_shot41", "X_shot47", "X_shot52"]]
        # [Y_keys.remove(i) for i in ["Y_shot41", "Y_shot47", "Y_shot52"]]

        for k in X_keys:
            x, y = f[k].shape
            if y < 5:
                continue
            xnew = f[k][0:x//2*2, 0:y//2*2]
            xnew = xnew[..., np.newaxis]
            X.append(xnew)
        for k in Y_keys:
            x, y = f[k].shape
            if y < 5:
                continue
            ynew = f[k][0:x//2*2, 0:y//2*2]
            ynew = np.stack([ynew, 1 - ynew], axis=-1)
            Y.append(ynew)
    return X, Y


folder = "/home/cougarnet.uh.edu/pyuan2/Ben PC/PycharmProjects/2018 Fall/Seismic/Project_report/03.27.19/Paper/"
fout = folder + "vec_figures_paper/"

file1 = "prob_def.pickle"
output = "problem_enhance"
# x_1, y_1, edge_1 = loadPickle(folder + file1)
# plot_figure_1(x_1, y_1, edge_1, fout+output, clipval=5)

## name = "/home/cougarnet.uh.edu/pyuan2/Ben PC/PycharmProjects/2018 Fall/Seismic/Project_report/03.27.19/Paper/LL120.pickle"
## savePickle(name, [tX, tY, pX, figurePath, idx, Y_label, Y_prediction])

file1_1 = "LLNoisy120.pickle"
file1_2 = "LLDisc50.pickle"
file2_1 = "CENoisy120.pickle"
file2_2 = "CEDisc50.pickle"

X, Y = readTest("/home/cougarnet.uh.edu/pyuan2/Ben PC/PycharmProjects/2018 Fall/Seismic/syn_fbk_example/pickingNoisy.h5")
x120 = X[120,1:1249, :, 0]
#
plot_figure_2(file1_1, (7, 5), fout, "LLNoisy120", x120, clipval=10)
plot_figure_2(file2_1, (7, 5), fout, "CENoisy120", x120, clipval=10)
#
# plot_figure_2(file1_2, (5, 5), fout, "LLDisc50")
# plot_figure_2(file2_2, (5, 5), fout, "CEDisc50")


file3_1 = "FPP46.pickle"
file3_2 = "NPP46.pickle"

# X, Y = readField("/home/cougarnet.uh.edu/pyuan2/Ben PC/PycharmProjects/2018 Fall/Seismic/syn_fbk_example/concatField_test.h5")
# x46 = X[46]
#
# plot_figure_3(file3_1, file3_2, (6, 6), fout, x46, clipval=5)


print("")


