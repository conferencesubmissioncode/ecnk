import utils
import config
import numpy as np
import cupy as cp

X_train, X_test = utils.load_cifar10("data") #if config.fargs['dataset'] == "cifar10" else utils.load_cifar100("data")
X_train = (X_train / 255.0).astype(np.float32) 
X_test = (X_test / 255.0).astype(np.float32) 
mean = X_train.mean(axis = (0, 2, 3))
std = X_train.std(axis = (0, 2, 3))
X_train = (X_train - mean[:, None, None]) / std[:, None, None]
X_test = (X_test - mean[:, None, None]) / std[:, None, None]
X_train_flip = np.flip(X_train, 3)
X_test_flip = np.flip(X_test, 3)

X_train = cp.pad(cp.asarray(X_train).reshape(50000, 3, 32, 32), ((0, 0), (0, 0), (1, 1), (1, 1)), mode = "constant").reshape(50000, 3, 34 * 34)
X_test = cp.pad(cp.asarray(X_test).reshape(10000, 3, 32, 32), ((0, 0), (0, 0), (1, 1), (1, 1)), mode = "constant").reshape(10000, 3, 34 * 34)
X_train_flip = cp.pad(cp.asarray(X_train_flip).reshape(50000, 3, 32, 32), ((0, 0), (0, 0), (1, 1), (1, 1)), mode = "constant").reshape(50000, 3, 34 * 34)
X_test_flip = cp.pad(cp.asarray(X_test_flip).reshape(10000, 3, 32, 32), ((0, 0), (0, 0), (1, 1), (1, 1)), mode = "constant").reshape(10000, 3, 34 * 34)

def featurize(st, ed, train, flip = False):
    if flip:
        return X_train_flip[st:ed] if train else X_test_flip[st:ed] 
    else:
        return X_train[st:ed] if train else X_test[st:ed]
