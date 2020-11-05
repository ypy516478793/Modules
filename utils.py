from sklearn.model_selection import train_test_split
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from PIL import Image
import h5py

def padding(img, bar=20, mirror=False):
    """
    1 layer: bar=8
    2 layers: bar=20
    3 layers: bar=44
    """
    if len(img.shape) == 2:
        print("Images mirror should be processed in batches")
    if mirror:
        img = np.concatenate([img[:,:,bar-1::-1,...], img, img[:,:,:-bar-1:-1,...]], axis=2)
        img = np.concatenate([img[:,bar-1::-1,:,...], img, img[:,:-bar-1:-1,:,...]], axis=1)
    else:
        if img.shape == 3:
            img = np.pad(img, ((0, 0), (bar, bar), (bar, bar)), 'constant')
        else:
            img = np.pad(img, ((0, 0), (bar, bar), (bar, bar), (0, 0)), 'constant')
    return img

def readTest(fileName, norm):
    with h5py.File(fileName, "r") as f:
        if fileName == "picking.h5" or fileName == "pickingBnd.h5":
            X = f['X_data'][:2000:10]                       #modify for other dataset !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            Y = f['Y_data'][:2000:10]                       #modify for other dataset !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        else:
            X = f['X_data']
            Y = f['Y_data']
        x, y, z = X.shape
        X = X[:, 0:y//2*2, 0:z//2*2]
        Y = Y[:, 0:y//2*2, 0:z//2*2]
        X = normalization(X, norm)
        Y = np.stack([Y, 1-Y], axis=-1)
        X = X[..., np.newaxis]
    return X, Y

def readSegy(fileName, norm, clipval=None):
    with h5py.File(fileName, "r") as f:
        X = f['X_data'][:,:100,:,:]
        X = X.reshape([-1, 1250, 2001])
        X = normalization(X, norm)
        if clipval:
            for i, Xi in enumerate(X):
                X[i] = clip(Xi, clipval)
        # X = normalization(X, "divideMax")
        Y = np.zeros(shape=X.shape)
        x, y, z = X.shape
        X = X[:, 0:y//2*2, 0:z//2*2]
        Y = Y[:, 0:y//2*2, 0:z//2*2]
        Y[:, 0:y//2, :] = 1
        Y = np.stack([Y, 1-Y], axis=-1)
        X = X[..., np.newaxis]
    return X, Y

def readFootHill(fileName, norm):
    with h5py.File(fileName, "r") as f:
        X, Y = deque(), deque()
        X_keys = [key for key in list(f.keys()) if key[0] == 'X']
        X_keys.sort(key=lambda x: int(x[6:]))
        Y_keys = [key for key in list(f.keys()) if key[0] == 'Y']
        Y_keys.sort(key=lambda x: int(x[6:]))
        for k in X_keys:
            x, y = f[k].shape
            # xnew = f[k][0:x//2*2, 0:y//2*2]
            xnew = f[k][0:1200, 0:y // 2 * 2]
            if norm == "standardize":
                print("Standard normalization is not applicable for this data.")
            else:
                # xnew = normalization(xnew, norm)
                xnew = normalization(xnew[np.newaxis,:], norm)[0,...]
            xnew = xnew[..., np.newaxis]
            X.append(xnew)
        for k in Y_keys:
            x, y = f[k].shape
            # ynew = f[k][0:x//2*2, 0:y//2*2]
            ynew = f[k][0:1200, 0:y // 2 * 2]
            ynew = np.stack([ynew, 1 - ynew], axis=-1)
            Y.append(ynew)
    return X, Y

def clip(binarray, clipval=10):
    maxval = np.percentile(binarray, 100 - clipval / 2)
    minval = np.percentile(binarray, clipval / 2)
    binarray = np.clip(binarray, minval, maxval)
    return binarray

def MMnormal(x):
    a = 2 / (np.max(x) - np.min(x))
    b = (np.max(x) + np.min(x)) / (np.min(x) - np.max(x))
    x = a * x + b
    return x

def smoothTrianglesmoothT (data, degree, dropVals=False):
    triangle=np.array(list(range(degree)) + [degree] + list(range(degree)[::-1])) + 1
    smoothed=[]
    for i in range(degree, len(data) - degree * 2):
        point=data[i:i + len(triangle)] * triangle
        smoothed.append(sum(point)/sum(triangle))
    if dropVals:
        return smoothed
    smoothed=[smoothed[0]]*int(degree + degree/2) + smoothed
    while len(smoothed) < len(data):
        smoothed.append(smoothed[-1])
    return smoothed

def write_spec(path, args):
    file = open(path + '/config.txt', 'a')
    file.write('validAcc: '+'{:.2f}'.format(args[0])+'\n')
    file.write('normType: '+args[1]+'\n')
    file.write('test error rate: '+ '{:.2f}'.format(args[2]) +'\n')
    file.write('average picking error: '+'{:.2f}'.format(args[3])+'\n')
    file.write('mean square error: '+'{:.2f}'.format(args[4])+'\n')
    file.write('FAP error rate: '+'{:.2f}'.format(args[5])+'\n')
    file.close()

def DMnormal(x):
    return x / np.max(np.abs(x))

def normalization(X, norm):
    if norm == "noNorm":
        print("No normalization.")
    elif norm == "minMax":
        print("Min-Max normalization.")
        # X = np.asanyarray([MMnormal(X[i]) for i in range(len(X))])
        n, h, w = X.shape
        X = np.asanyarray([MMnormal(X[i, :, j]) for i in range(n) for j in range(w)])
        X = X.reshape([n, w, h])
        X = np.transpose(X, (0, 2, 1))
    elif norm == "standardize":
        print("Standard normalization.")
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        X = np.nan_to_num(X)
    elif norm == "divideMax":
        print("Divide Max normalization.")
        n, h, w = X.shape
        X = np.asanyarray([DMnormal(X[i, :, j]) for i in range(n) for j in range(w)])
        X = X.reshape([n, w, h])
        X = np.transpose(X, (0, 2, 1))
    else:
        print("Something wrong")
    return X

class BatchDataReader(object):
    def __init__(self, fileName, n_channel, n_class, addnoise=False, norm="noNorm"):
        if fileName == "picking.h5" or fileName == "pickingBnd.h5":
            with h5py.File(fileName, "r") as f:
                self.X = f['X_data'][:1000]
                self.Y = f['Y_data'][:1000]
            width = self.X.shape[2]
            height = self.X.shape[1]
            num = self.X.shape[0]
            cropWidth = 600
            cropHeight = height
            rdStart = np.random.randint(0, width-cropWidth, num)
            cropX = np.zeros([num, cropHeight, cropWidth])
            cropY = np.zeros([num, cropHeight, cropWidth])
            for i in range(num):
                cropX[i] = self.X[i, :, rdStart[i]: rdStart[i]+cropWidth]
                cropY[i] = self.Y[i, :, rdStart[i]: rdStart[i]+cropWidth]
            self.X = cropX
            self.Y = cropY
        else:
            with h5py.File(fileName, "r") as f:
                self.X = f['X_data'][:]
                self.Y = f['Y_data'][:]
            if fileName == "FBK_PICK.h5":
                self.X = np.concatenate([self.X[:50], self.X[215:]], axis=0)
                self.Y = np.concatenate([self.Y[:50], self.Y[215:]], axis=0)
        self.X = normalization(self.X, norm)

        if addnoise:
            self.X = add_noise(self.X)
        self.n_channel = n_channel
        self.n_class = n_class
        self._process_data()
        indices = np.arange(len(self.X))
        if fileName == "FBK_PICK.h5":
            self.X_train, self.X_test, self.Y_train, self.Y_test, self.idx_train, self.idx_test = train_test_split(
                self.Xnew, self.Ynew, indices, test_size=0.0, random_state=42)
        else:
            self.X_train, self.X_test, self.Y_train, self.Y_test, self.idx_train, self.idx_test = train_test_split(
                self.Xnew, self.Ynew, indices, test_size=0.4, random_state=42)
        self.pointer = 0


    def __call__(self, n):
        restart = (self.pointer + n) // len(self.X_train)
        pointer_end = (self.pointer + n) % len(self.X_train)
        if restart:
            x_batch = np.vstack([self.X_train[self.pointer:], self.X_train[:pointer_end]])
            y_batch = np.vstack([self.Y_train[self.pointer:], self.Y_train[:pointer_end]])
        else:
            x_batch = self.X_train[self.pointer: self.pointer+n]
            y_batch = self.Y_train[self.pointer: self.pointer+n]
        self.pointer = pointer_end

        return x_batch, y_batch

    def _process_data(self):
        # for i in range(len(self.X)):
        #     x, y = self.X[i], self.Y[i]
        #     imX = Image.fromarray(x)
        #     imY = Image.fromarray(y)
        #     self.Xnew[i] = np.array(imX.resize([self.size, self.size]))
        #     self.Ynew[i] = np.array(imY.resize([self.size, self.size]))
        x, y, z = self.X.shape
        self.Xnew = self.X[:, 0:y//2*2, 0:z//2*2]
        self.Ynew = self.Y[:, 0:y//2*2, 0:z//2*2]

        if self.n_class == 2:
            # self.Ynew = np.stack([1-self.Ynew, self.Ynew], axis=-1)
            self.Ynew = np.stack([self.Ynew, 1-self.Ynew], axis=-1)
        self.Xnew = self.Xnew[..., np.newaxis]

def add_noise(batch, mean=0, var=0.1, amount=0.1, mode='pepper'):
    original_size = batch.shape
    batch = np.squeeze(batch)
    batch_noisy = np.zeros(batch.shape)
    for ii in range(batch.shape[0]):
        image = np.squeeze(batch[ii])
        if mode == 'gaussian':
            gauss = np.random.normal(mean, var, image.shape)
            image = image + gauss
        elif mode == 'pepper':
            num_pepper = np.ceil(amount * image.size)
            coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
            image[coords] = 0
        elif mode == "s&p":
            s_vs_p = 0.5
            # Salt mode
            num_salt = np.ceil(amount * image.size * s_vs_p)
            coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
            image[coords] = 1
            # Pepper mode
            num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
            image[coords] = 0
        batch_noisy[ii] = image
    return batch_noisy.reshape(original_size)

if __name__ == '__main__':
    data_provider = BatchDataReader('FBK_PICK.h5', 1, 2, True, norm="noNorm")
    x, y = data_provider(2)
    print('')
