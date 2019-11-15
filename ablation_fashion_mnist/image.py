import utils
import config
import numpy as np
import cupy as cp

print "Loading CIFAR10" if config.fargs['dataset'] == "cifar10" else "Loading Fashion MNIST"
X_train, X_test = utils.load_cifar10("data") if config.fargs['dataset'] == "cifar10" else utils.load_fashion_mnist("data")
X_train = (X_train / 255.0).astype(np.float32) 
X_test = (X_test / 255.0).astype(np.float32) 
mean = X_train.mean(axis = (0, 2, 3))
std = X_train.std(axis = (0, 2, 3))
X_train = (X_train - mean[:, None, None]) / std[:, None, None]
X_test = (X_test - mean[:, None, None]) / std[:, None, None]
X_train_flip = np.flip(X_train, 3)
X_test_flip = np.flip(X_test, 3)
X_train = cp.asarray(X_train).reshape(X_train.shape[0], X_train.shape[1], config.pixel * config.pixel)
X_test = cp.asarray(X_test).reshape(X_test.shape[0], X_test.shape[1], config.pixel * config.pixel)
X_train_flip = cp.asarray(X_train_flip).reshape(X_train_flip.shape[0], X_train_flip.shape[1], config.pixel * config.pixel)
X_test_flip = cp.asarray(X_test_flip).reshape(X_test_flip.shape[0], X_test_flip.shape[1], config.pixel * config.pixel)

def featurize(st, ed, train, flip = False):
    if flip:
        return X_train_flip[st:ed] if train else X_test_flip[st:ed] 
    else:
        return X_train[st:ed] if train else X_test[st:ed]
