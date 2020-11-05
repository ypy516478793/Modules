from scipy.spatial.distance import cdist
import numpy as np
import itertools
import random

def hamming_set(num_crops, num_permutations, selection, output_file_name):
    """
    generate and save the hamming set
    :param num_crops: number of tiles from each image
    :param num_permutations: Number of permutations to select (i.e. number of classes for the pretext task)
    :param selection: Sample selected per iteration based on hamming distance: [max] highest; [mean] average
    :param output_file_name: name of the output HDF5 file
    """
    P_hat = np.array(list(itertools.permutations(list(range(num_crops)), num_crops)))
    n = P_hat.shape[0]

    for i in range(num_permutations):
        if i == 0:
            j = np.random.randint(n)
            P = np.array(P_hat[j]).reshape([1, -1])
        else:
            P = np.concatenate([P, P_hat[j].reshape([1, -1])], axis=0)

        P_hat = np.delete(P_hat, j, axis=0)
        D = cdist(P, P_hat, metric='hamming').mean(axis=0).flatten()

        if selection == 'max':
            j = D.argmax()
        elif selection == 'mean':
            m = int(D.shape[0] / 2)
            S = D.argsort()
            j = S[np.random.randint(m - 10, m + 10)]


    hamfile = './hamming_set/' + output_file_name + str(num_permutations) + '.npz'
    np.savez_compressed(hamfile,  data=P)
    print('file created --> ' + output_file_name + str(num_permutations) + '.npz')


def one_hot(self, y):
    """
    Explain
    """
    return np.array([[1 if y[i] == j else 0 for j in range(self.numClasses)] for i in range(y.shape[0])])


def __data_generation_normalize(self, x):
    """
    Explain
    """
    x -= meanTensor
    x /= stdTensor
    # This implementation modifies each image individually
    y = np.empty(batchSize)
    # Python list of 4D numpy tensors for each channel
    X = [np.empty((batchSize, tileSize, tileSize, numChannels), np.float32)
         for _ in range(numCrops)]
    for image_num in range(batchSize):
        # Transform the image into its nine crops
        single_image, y[image_num] = create_croppings(x[image_num])
        for image_location in range(numCrops):
            X[image_location][image_num, :, :, :] = single_image[:, :, :, image_location]
    return X, y


def generate(self, mode='train'):
    """
    Explain
    """
    if mode == 'train':
        h5f = h5py.File(data_path, 'r')
        x = h5f['train_img'][batchIndexTrain * batchSize:(batchIndexTrain + 1) * batchSize, ...]
        h5f.close()
        if numChannels == 1:
            x = np.expand_dims(x, axis=-1)
        X, y = __data_generation_normalize(x.astype(np.float32))
        batchIndexTrain += 1  # Increment the batch index
        if batchIndexTrain == numTrainBatch:
            batchIndexTrain = 0
    elif mode == 'valid':
        h5f = h5py.File(data_path, 'r')
        x = h5f['val_img'][batchIndexVal * batchSize:(batchIndexVal + 1) * batchSize, ...]
        h5f.close()
        if numChannels == 1:
            x = np.expand_dims(x, axis=-1)
        X, y = __data_generation_normalize(x.astype(np.float32))
        batchIndexVal += 1  # Increment the batch index
        if batchIndexVal == numValBatch:
            batchIndexVal = 0
    elif mode == 'test':
        h5f = h5py.File(data_path, 'r')
        x = h5f['test_img'][batchIndexTest * batchSize:(batchIndexTest + 1) * batchSize, ...]
        h5f.close()
        if numChannels == 1:
            x = np.expand_dims(x, axis=-1)
        X, y = __data_generation_normalize(x.astype(np.float32))
        batchIndexTest += 1  # Increment the batch index
        if batchIndexTest == numTestBatch:
            batchIndexTest = 0
    return np.transpose(np.array(X), axes=[1, 2, 3, 4, 0]), one_hot(y)


def create_croppings_copy(image, cropSize, numClasses, tileSize, numChannels, cellSize, maxHammingSet):
    """
    Take in a 3D numpy array (256x256x3) and a 4D numpy array containing 9 "jigsaw" puzzles.
    Dimensions of the output array is 64 (height) x 64 (width) x 3 (colour channels) x 9(each cropping)

    The 3x3 grid is numbered as follows:
    0    1    2
    3    4    5
    6    7    8
    """
    # Jitter the colour channel
    # image = color_channel_jitter(image)

    y_dim, x_dim = image.shape[:2]
    # Have the x & y coordinate of the crop
    if x_dim != cropSize:
        crop_x = random.randrange(x_dim - cropSize)
        crop_y = random.randrange(y_dim - cropSize)
    else:
        crop_x, crop_y = 0, 0

    # Select which image ordering we'll use from the maximum hamming set
    perm_index = random.randrange(numClasses)
    final_crops = np.zeros((tileSize, tileSize, numChannels, numCrops), dtype=np.float32)
    n_crops = int(np.sqrt(numCrops))
    for row in range(n_crops):
        for col in range(n_crops):
            x_start = crop_x + col * cellSize + random.randrange(cellSize - tileSize)
            y_start = crop_y + row * cellSize + random.randrange(cellSize - tileSize)
            # Put the crop in the list of pieces randomly according to the number picked
            final_crops[:, :, :, maxHammingSet[perm_index, row * n_crops + col]] = \
                image[y_start:y_start + tileSize, x_start:x_start + tileSize, :]
    return final_crops, perm_index


def create_croppings(image, numCrops, maxHammingSet, tileSize=75):
    """
    Take in a 3D numpy array (256x256x3) and a 4D numpy array containing 9 "jigsaw" puzzles.
    Dimensions of the output array is 64 (height) x 64 (width) x 3 (colour channels) x 9(each cropping)

    The 3x3 grid is numbered as follows:
    0    1    2
    3    4    5
    6    7    8
    """
    # Jitter the colour channel
    # image = color_channel_jitter(image)

    numChannels, y_dim, x_dim = image.shape
    # Have the x & y coordinate of the crop
    # if x_dim != cropSize:
    #     crop_x = random.randrange(x_dim - cropSize)
    #     crop_y = random.randrange(y_dim - cropSize)
    # else:
    #     crop_x, crop_y = 0, 0

    # Select which image ordering we'll use from the maximum hamming set
    numClasses = len(maxHammingSet)
    perm_index = random.randrange(numClasses)
    final_crops = np.zeros((numChannels, tileSize, tileSize, numCrops), dtype=np.float32)
    n_crops = int(np.sqrt(numCrops))
    for row in range(n_crops):
        for col in range(n_crops):
            x_start = col * tileSize
            y_start = row * tileSize
            # Put the crop in the list of pieces randomly according to the number picked
            final_crops[:, :, :, maxHammingSet[perm_index, row * n_crops + col]] = \
                image[:, y_start:y_start + tileSize, x_start:x_start + tileSize]
    # assign back
    shuffle_img = np.zeros((numChannels, x_dim, y_dim), dtype=np.float32)
    for row in range(n_crops):
        for col in range(n_crops):
            x_start = col * tileSize
            y_start = row * tileSize
            shuffle_img[:, y_start:y_start + tileSize, x_start:x_start + tileSize] = \
                final_crops[:, :, :, row * n_crops + col]
    return final_crops, shuffle_img, perm_index

if __name__ == '__main__':
    # hamming_set(num_crops=9,
    #             num_permutations=100,
    #             selection='max',
    #             output_file_name='max_hamming_set_')
    
    hammingfile = "max_hamming_set_100.npz"
    hammingSet = np.load(hammingfile)["data"]
    image = np.load("ctimg.npz")["data"]
    final_crops, shuffle_img, perm_index = create_croppings(image, numCrops=9, maxHammingSet=hammingSet, tileSize=75)

    
    print("")