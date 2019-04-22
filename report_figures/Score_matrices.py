from matplotlib.patches import Circle
from skimage import io

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import pandas as pd
import numpy as np
import pickle

fileDict = {
"double_4c": "/home/cougarnet.uh.edu/pyuan2/Projects2019/keras-resnet/results/resnet18_flyCells_multiTask_weighted_doubleRNAi_6fold/",
"double_2c": "/home/cougarnet.uh.edu/pyuan2/Projects2019/keras-resnet/results/resnet18_flyCells_multiTask_weighted_doubleRNAi_OnlyNuclei_6fold/",
"single_4c": "/home/cougarnet.uh.edu/pyuan2/Projects2019/keras-resnet/results/resnet18_flyCells_multiTask_weighted_singleRNAi_6fold/",
"single_2c": "/home/cougarnet.uh.edu/pyuan2/Projects2019/keras-resnet/results/resnet18_flyCells_multiTask_weighted_singleRNAi_OnlyNuclei_6fold/",
}

sqrMat = {}

for key in fileDict:
    pickleFile = fileDict[key] + "All_train/" + key + "_square_matrix.pickle"
    with open(pickleFile, 'rb') as f:
        sqrMat[key], labelDict, wellInfo = pickle.load(f)

fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
loc = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]

for i, key in enumerate(fileDict):
    data = pd.DataFrame(data=sqrMat[key], index=labelDict.keys(), columns=labelDict.keys())
    sns.heatmap(data, ax=loc[i], cmap="Blues")
    loc[i].set_title(key)

plt.show()
print("")

# def one_hot(x, K):
#     return np.array(x[:, None] == np.arange(K)[None, :], dtype=int)
#
# pfile = "/home/cougarnet.uh.edu/pyuan2/Projects2019/keras-resnet/results/resnet18_flyCells_multiTask_weighted_doubleRNAi_OnlyNuclei_6fold/0/data.pickle"
# with open(pfile, 'rb') as f:
#     y_train, y_score = pickle.load(f)
# y_train = one_hot(y_train, 17).squeeze()

# x = range(len(y_score[:,6]))
# sns.lineplot(x = x, y = y_score[:, 6])
# plt.title("Probability of being EMPTY")
# plt.savefig("/home/cougarnet.uh.edu/pyuan2/Projects2019/Fly_Cell_DeepLearning/Prob_empty.png", dpi=300, bbox_inches='tight')

print("")