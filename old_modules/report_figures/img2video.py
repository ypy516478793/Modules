from natsort import natsorted

import glob
import cv2

img_array = []
for filename in natsorted(glob.glob('/home/cougarnet.uh.edu/pyuan2/Projects2019/maml/logs/sine/cls_5.mbs_25.ubs_10.numstep1.updatelr0.01nonorm.mt70000kp0.90/amp3.00_ph0.00_pts2/*.png')):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)

out = cv2.VideoWriter('/home/cougarnet.uh.edu/pyuan2/Projects2019/maml/logs/sine/cls_5.mbs_25.ubs_10.numstep1.updatelr0.01nonorm.mt70000kp0.90/amp3.00_ph0.00_pts2/project.avi', cv2.VideoWriter_fourcc(*'DIVX'), 4, size)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()

from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

loadFolder = "/home/cougarnet.uh.edu/pyuan2/Ben PC/PycharmProjects/2018 Fall/Seismic/fbk_training_data_with_label_r02/"
saveFolder = os.path.join(loadFolder, "commonReceiver/")
with open(os.path.join(saveFolder, "lineArray.pkl"), "rb") as f:
    lineArray = pickle.load(f)


## Only plot ##
plt.figure(figsize=(12, 8))
plt.ylim([870000, 910000])
plt.xlim([967500, 1002500])
plt.ion()
batchSize = 100
color = 2
num_traces = 10000
# num_traces = len(lineArray)
for i in tqdm(range(0, num_traces, batchSize)):
    sx_batch = lineArray[i: i+batchSize, 1]
    sy_batch = lineArray[i: i+batchSize, 2]
    plt.plot(sx_batch, sy_batch, "*r")
    if np.all(lineArray[i, 1:3] == lineArray[i-1, 1:3]):
        if len(np.unique(sx_batch)) == 1 and len(np.unique(sy_batch)) == 1:
            plt.plot(lineArray[i: i + batchSize, 4], lineArray[i: i + batchSize, 5], ".C{:d}".format(color))
        else:
            changeIdx = np.min([np.min(np.where(sx_batch != sx_batch[0])), np.min(np.where(sy_batch != sy_batch[0]))])
            plt.plot(lineArray[i: i + changeIdx, 4], lineArray[i: i + changeIdx, 5], ".C{:d}".format(color))
            color = 2 - color
            plt.plot(lineArray[i + changeIdx: i + batchSize, 4], lineArray[i + changeIdx: i + batchSize, 5], ".C{:d}".format(color))
    else:
        color = 2 - color
        plt.plot(lineArray[i: i + batchSize, 4], lineArray[i: i + batchSize, 5],".C{:d}".format(color))

    # plt.plot(lineArray[i: i+batchSize, 4], lineArray[i: i+batchSize, 5], ".C1")
    plt.draw()
    plt.pause(0.00001)



## Plot and save to mp4 ##
pointColor = 2
batchSize = 100

def animate(i):
    # for i in tqdm(range(0, num_traces, batchSize)):
    global pointColor
    i = i * batchSize
    sx_batch = lineArray[i: i + batchSize, 1]
    sy_batch = lineArray[i: i + batchSize, 2]
    plt.plot(sx_batch, sy_batch, "*r")
    if np.all(lineArray[i, 1:3] == lineArray[i - 1, 1:3]):
        if len(np.unique(sx_batch)) == 1 and len(np.unique(sy_batch)) == 1:
            plt.plot(lineArray[i: i + batchSize, 4], lineArray[i: i + batchSize, 5], ".C{:d}".format(pointColor))
        else:
            changeIdx = np.min(
                [np.min(np.where(sx_batch != sx_batch[0])), np.min(np.where(sy_batch != sy_batch[0]))])
            plt.plot(lineArray[i: i + changeIdx, 4], lineArray[i: i + changeIdx, 5], ".C{:d}".format(pointColor))
            pointColor = 2 - pointColor
            plt.plot(lineArray[i + changeIdx: i + batchSize, 4], lineArray[i + changeIdx: i + batchSize, 5],
                     ".C{:d}".format(pointColor))
    else:
        pointColor = 2 - pointColor
        plt.plot(lineArray[i: i + batchSize, 4], lineArray[i: i + batchSize, 5], ".C{:d}".format(pointColor))
    # plt.plot(lineArray[i: i+batchSize, 4], lineArray[i: i+batchSize, 5], ".C1")
    # plt.draw()
    # plt.pause(0.00001)

fig = plt.figure(figsize=(12, 8))
plt.ylim([870000, 910000])
plt.xlim([967500, 1002500])

# anim = FuncAnimation(fig, animate, frames=1000, fargs=(color, batchSize), interval=1, blit=False)
anim = FuncAnimation(fig, animate, frames=1000, interval=1, blit=False, repeat=False)
# plt.show()

writervideo = FFMpegWriter(fps=30)
anim.save(os.path.join(saveFolder, "animation.mp4"), writer=writervideo)

print("Save animation to: " + os.path.join(saveFolder, "animation.mp4"))

# Change global variable inside the local function: https://www.geeksforgeeks.org/global-keyword-in-python/

