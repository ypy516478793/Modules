import numpy as np
import pickle
import h5py


def saveH5(fileName, X, Y):
    with h5py.File(fileName, 'w') as f:
        f.create_dataset('X_data', data=X)
        f.create_dataset('Y_data', data=Y)

def loadH5(fileName):
    with h5py.File(fileName, 'r') as f:
        X = f['X_data'][:]
        Y = f['Y_data'][:]
    return X, Y

def saveListH5(fileName, X, Y):
    with h5py.File(fileName, "w") as f:
        for n, d in enumerate(X):
            f.create_dataset(name='X_shot{:d}'.format(n), data=d)
        for n, d in enumerate(Y):
            f.create_dataset(name='Y_shot{:d}'.format(n), data=d)

def readListH5(fileName):
    with h5py.File(fileName, "r") as f:
        X, Y = [], []
        X_keys = [key for key in list(f.keys()) if key[0] == 'X']
        X_keys.sort(key=lambda x: int(x[6:]))
        Y_keys = [key for key in list(f.keys()) if key[0] == 'Y']
        Y_keys.sort(key=lambda x: int(x[6:]))
        for k in X_keys:
            X.append(f[k][:])
        for k in Y_keys:
            Y.append(f[k][:])
    return X, Y

def savePickle(name, data):
    with open(name, 'wb') as f:
        pickle.dump(data, f)

def loadPickle(name):
    with open(name, 'rb') as f:
        data = pickle.load(f)
    return data

if __name__ == '__main__':

    # X = np.arange(12).reshape((3, 4))
    # Y = np.random.randint(0, 2, (3, 4))
    # saveH5('picking.h5', X, Y)

    X, Y = readListH5('/home/cougarnet.uh.edu/pyuan2/Ben PC/PycharmProjects/2018 Fall/Seismic/syn_fbk_example/pickingDiscnt.h5')
    print('')