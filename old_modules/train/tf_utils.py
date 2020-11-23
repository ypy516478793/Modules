from sklearn.model_selection import train_test_split
import numpy as np

def one_hot(x, K):
    return np.array(x[:, None] == np.arange(K)[None, :], dtype=int)


def normalization(X, norm):
    if norm == "noNorm":
        print("No normalization.")
    elif norm == "standardize":
        print("Standard normalization.")
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        X = np.nan_to_num(X)
    elif norm == "minMax":
        print("Min-Max normalization to [0, 1].")
        n, h, w = X.shape
        MMnorm = lambda x: (x - np.min(x)) /  (np.max(x) - np.min(x))
        X = np.asanyarray([MMnorm(X[i, :, j]) for i in range(n) for j in range(w)])
        X = X.reshape([n, w, h])
        X = np.transpose(X, (0, 2, 1))
    elif norm == "minMax_sym":
        print("Min-Max normalization to [-1, 1].")
        n, h, w = X.shape
        MMnorm_sym = lambda x: (2 * x - np.max(x) - np.min(x)) /  (np.max(x) - np.min(x))
        X = np.asanyarray([MMnorm_sym(X[i, :, j]) for i in range(n) for j in range(w)])
        X = X.reshape([n, w, h])
        X = np.transpose(X, (0, 2, 1))
    elif norm == "divideMax":
        print("Divide Max normalization to [-1, 1].")
        n, h, w = X.shape
        DMnorm = lambda x: x / np.max(np.abs(x))
        X = np.asanyarray([DMnorm(X[i, :, j]) for i in range(n) for j in range(w)])
        X = X.reshape([n, w, h])
        X = np.transpose(X, (0, 2, 1))
    else:
        print("Something wrong")
    return X

def splitData(X, Y):
    indices = np.arange(len(X))
    X_train, X_test, Y_train, Y_test, idx_train, idx_test = train_test_split(
        X, Y, indices, test_size=0.4, random_state=42)
    return X_train, X_test, Y_train, Y_test, idx_train, idx_test

if __name__ == '__main__':
    x = np.random.randint(0, 5, (10))
    print(x)
    print(one_hot(x, 5))