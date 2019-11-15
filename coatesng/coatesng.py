import numpy as np
import cupy as cp
import chainer.functions as F
import utils
import config

X_train, X_test = utils.load_cifar10("data") if config.fargs['dataset'] == "cifar10" else utils.load_cifar100("data")

class CoatesNgNet:
    def __init__(self):
        self.bias = config.fargs['bias']
        all_patches, idxs = utils.grab_patches(X_train, patch_size = config.fargs['patch_size'], max_threads = 16, tot_patches = 100000)
        all_patches = utils.normalize_patches(all_patches, zca_bias = 1e-3)
        filters = all_patches[np.random.choice(all_patches.shape[0], config.fargs['num_filters'], replace = False)].astype(np.float32)
        self.filters = cp.asarray(filters)
        if config.flip:
            self.filters = cp.concatenate((self.filters, cp.flip(self.filters, 3)), axis = 0)        
            
    def forward(self, x):
        n = x.shape[0]
        c = self.filters.shape[0]
        res = cp.zeros((n, c * 2, config.pixel, config.pixel), dtype = cp.float32)
        
        res[:, c:] = F.convolution_2d(x, self.filters, pad = config.fargs['pad_size']).data if 'pad_size' in config.fargs else F.convolution_2d(x, self.filters).data
        res[:, :c] = res[:, c:]
        
        cp.add(res[:, :c], -self.bias, res[:, :c])
        cp.maximum(res[:, :c] , 0.0, res[:, :c])
        #res[:, :c] = cp.maximum(res[:, c:] - self.bias, 0.0)
        
        cp.multiply(res[:, c:], -1.0, res[:, c:])
        cp.add(res[:, c:], -self.bias, res[:, c:])
        cp.maximum(res[:, c:] , 0.0, res[:, c:])
        #res[:, c:] = cp.maximum(-res[:, c:] - self.bias, 0.0)
        
        return res.reshape(n, c * 2, -1)

net = CoatesNgNet()

X_train = X_train.astype(np.float32) / 255.0
X_test = X_test.astype(np.float32)  / 255.0
X_train_flip = np.flip(X_train, 3)
X_test_flip = np.flip(X_test, 3)

X_train = cp.asarray(X_train)
X_test = cp.asarray(X_test)
X_train_flip = cp.asarray(X_train_flip)
X_test_flip = cp.asarray(X_test_flip)

def featurize(st, ed, train, flip = False):
    if flip:
        return net.forward(X_train_flip[st:ed] if train else X_test_flip[st:ed])
    else:
        return net.forward(X_train[st:ed] if train else X_test[st:ed])
