from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import pickle
import os



def write_results(fileName, actual, predicted, label, i=None):
    with open(fileName, 'a') as file:
        if i is not None:
            print('-------- Class {:0} -------'.format(i))
            file.write('-------- Class {:0} -------\n'.format(i))
            file.write("Gene: " + label + "\n")
        results = confusion_matrix(actual, predicted)
        file.write('Confusion Matrix :\n')
        file.write(np.array_str(results) + '\n')
        file.write('Accuracy Score :' + str(accuracy_score(actual, predicted)) + '\n')
        file.write('Report : \n')
        file.write(classification_report(actual, predicted) + '\n')

def get_confusion_matrix(actual, predicted):
    results = confusion_matrix(actual, predicted)
    print('Confusion Matrix :')
    print(results)
    print('Accuracy Score :', accuracy_score(actual, predicted))
    print('Report : ')
    print(classification_report(actual, predicted))

def loadPickle(folder):
    with open(folder + 'data.pickle', 'rb') as f:
        prediction = pickle.load(f)
    return prediction

def savePickle(folder, data):
    with open(folder + '_square_matrix.pickle', 'wb') as f:
        pickle.dump(data, f)

def checkFolder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

def one_hot(x, K):
    return np.array(x[:, None] == np.arange(K)[None, :], dtype=int)

def get_score_matrix(rootFolder, strChnl, labelDict, wellInfo, n_fold=10):

    mats = []
    labels = []
    for i in range(n_fold):
        file = os.path.join(rootFolder, str(i), "data.pickle")
        with open(file, "rb") as f:
            x_test, label, mat = pickle.load(f)
            # label, mat = pickle.load(f)
            labels.append(label)
            mats.append(mat)


    Y_score = np.concatenate(mats, axis=0)
    nb_classes = Y_score.shape[-1]
    gt = np.concatenate(labels, axis=0)
    Y_test = one_hot(gt, nb_classes).squeeze()
    logics = Y_score >= 0.5
    Y_pred = logics.astype(np.int)


    resultFolder = rootFolder + "/All/"
    checkFolder(resultFolder)

    sqrMat = np.zeros([nb_classes, nb_classes])
    for i in range(nb_classes):
        sqrMat[i, :] = np.mean(Y_score[np.where(gt==i)[0]], axis=0)

    # X, y, labelDict, wellInfo = loadPickle('../../')

    # import pickle
    # fileName = "double_4c_square_matrix.pickle" # can be changed to other file
    # with open(fileName, 'rb') as f:
    #     sqrMat, labelDict, wellInfo = pickle.load(f)

    savePickle(resultFolder + strChnl, [sqrMat, labelDict, wellInfo])

    plt.figure()
    plt.imshow(sqrMat)
    plt.title("Before biclustering")
    plt.savefig(resultFolder + "Before biclustering", dpi=300, bbox_inches='tight')
    plt.close()


    n_clusters = 5
    from sklearn.cluster.bicluster import SpectralCoclustering
    model_cluster = SpectralCoclustering(n_clusters=n_clusters, random_state=0)
    model_cluster.fit(sqrMat)
    fit_data = sqrMat[np.argsort(model_cluster.row_labels_)]
    fit_data = fit_data[:, np.argsort(model_cluster.column_labels_)]
    plt.matshow(fit_data)
    plt.title("After biclustering, n_cluster = {:d}".format(n_clusters))
    plt.savefig(resultFolder + "{:d} biclustering".format(n_clusters), dpi=300, bbox_inches='tight')
    plt.close()


    Y_test, Y_score, Y_pred = Y_test.T, Y_score.T, Y_pred.T


    invLabelDict = {}
    for key in labelDict:
        invLabelDict[labelDict[key]] = key

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(nb_classes):
        fpr[i], tpr[i], _ = roc_curve(Y_test[i], Y_score[i])
        roc_auc[i] = auc(fpr[i], tpr[i])

        plt.figure()
        lw = 2
        plt.plot(fpr[i], tpr[i], color='darkorange',
                 lw=lw, label='ROC curve (area = %0.4f)' % roc_auc[i])
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC for gene ' + invLabelDict[i])
        plt.legend(loc="lower right")
        plt.savefig(resultFolder + "roc gene: " + invLabelDict[i], dpi=300, bbox_inches='tight')
        plt.close()

    for i in range(nb_classes):
        print('-------- Class {:0} -------'.format(i))
        print("gene: " + invLabelDict[i])
        get_confusion_matrix(Y_test[i], Y_pred[i])
        write_results(resultFolder + 'output.txt', Y_test[i], Y_pred[i], invLabelDict[i], i)


    ordered_roc_auc = {}
    for key in roc_auc:
        ordered_roc_auc[invLabelDict[key]] = float("{0:.4f}".format(roc_auc[key]))
    ordered_roc_auc = sorted(ordered_roc_auc.items(), key=lambda x: -x[1])

    with open(resultFolder + 'results.txt', 'a') as file:
        file.write("ROC ranking: \n")
        file.write("{:25s} {:25s} {:4s} \n".format("Knocked Gene", "avg score", "Rank"))
        rank = 1
        for i in range(len(ordered_roc_auc)):
            line = "{:25s} {:25s} {:4s} \n".format(ordered_roc_auc[i][0], str(ordered_roc_auc[i][1]), str(rank))
            file.write(line)
            rank += 1
        # file.write(np.array_str(np.array(ordered_roc_auc)) +"\n")

root = "/home/cougarnet.uh.edu/pyuan2/Projects2019/keras-resnet/results/diff_Genes/"

row_data_pickle = "/home/cougarnet.uh.edu/pyuan2/Projects2019/Fly_Cell_DeepLearning/Fly_Cell_DeepLearning/All_datasets/Single_RNAi/Single_RNAi_data.pickle"
with open(row_data_pickle, 'rb') as f:
    X, y, labelDict, wellInfo = pickle.load(f)

# priGeneDict = {}
# folderList = os.listdir(root)
# folderList = [fd for fd in folderList if os.path.isdir(os.path.join(root, fd))]
# for i, pGene in enumerate(folderList):
#     priGeneDict[pGene] = i
# labelDict = priGeneDict

# labelDict = {}
# dataFolder = "/home/cougarnet.uh.edu/pyuan2/Projects2019/Fly_Cell_DeepLearning/Fly_Cell_DeepLearning/All_datasets/"
# folderList = os.listdir(dataFolder)
# folderList = [fd for fd in folderList if os.path.isdir(os.path.join(dataFolder, fd))]
# for i, pGene in enumerate(folderList):
#     labelDict[pGene] = i

chlDict = {"resnet18_flyCells_multiTask_weighted_6fold": "4c",
           "resnet18_flyCells_multiTask_weighted_OnlyNuclei_6fold": "2c"}
fileDict = {}
for key in chlDict:
    fileDict[chlDict[key]] = key

primFolderList = os.listdir(root)
primFolderList = [fd for fd in primFolderList if os.path.isdir(os.path.join(root, fd))]

for primFolder in primFolderList:
    primPath = os.path.join(root, primFolder)

    folderList = os.listdir(primPath)
    folderList = [fd for fd in folderList if os.path.isdir(os.path.join(primPath, fd))]

    sqrMat = {}
    for folder in folderList:
        folderPath = os.path.join(primPath, folder)
        get_score_matrix(folderPath, chlDict[folder], labelDict, wellInfo, n_fold=6)
        pickleFile = folder + "/All/" + chlDict[folder] + "_square_matrix.pickle"
        pickleFilePath = os.path.join(primPath, pickleFile)
        with open(pickleFilePath, 'rb') as f:
            sqrMat[chlDict[folder]], labelDict, wellInfo = pickle.load(f)


    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharex=True, sharey=True)
    loc = [axes[0], axes[1]]

    for i, key in enumerate(fileDict):
        label = [s.replace("_Double_RNAi", "") for s in labelDict.keys()]
        data = pd.DataFrame(data=sqrMat[key], index=label, columns=label)
        sns.heatmap(data, ax=loc[i], cmap="Blues", annot=True, fmt=".2f", annot_kws={"size": 5})
        loc[i].set_title(key)

    plt.savefig(primPath + "/Score_matrices_annot.png", dpi=300, bbox_inches='tight')





# fileDict = {
# "4c": "/home/cougarnet.uh.edu/pyuan2/Projects2019/keras-resnet/results/resnet18_flyCells_multiTask_weighted_doubleRNAi_6fold/",
# "double_2c": "/home/cougarnet.uh.edu/pyuan2/Projects2019/keras-resnet/results/resnet18_flyCells_multiTask_weighted_doubleRNAi_OnlyNuclei_6fold/",
# "single_4c": "/home/cougarnet.uh.edu/pyuan2/Projects2019/keras-resnet/results/resnet18_flyCells_multiTask_weighted_singleRNAi_6fold/",
# "single_2c": "/home/cougarnet.uh.edu/pyuan2/Projects2019/keras-resnet/results/resnet18_flyCells_multiTask_weighted_singleRNAi_OnlyNuclei_6fold/",
# }
#
# sqrMat = {}
#
# for key in fileDict:
#     pickleFile = fileDict[key] + "All/" + key + "_square_matrix.pickle"
#     with open(pickleFile, 'rb') as f:
#         sqrMat[key], labelDict, wellInfo = pickle.load(f)
#
# fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
# loc = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]
#
# for i, key in enumerate(fileDict):
#     data = pd.DataFrame(data=sqrMat[key], index=labelDict.keys(), columns=labelDict.keys())
#     sns.heatmap(data, ax=loc[i], cmap="Blues")
#     loc[i].set_title(key)
#
# plt.show()
# print("")