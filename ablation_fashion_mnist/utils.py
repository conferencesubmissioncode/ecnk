import concurrent.futures as fs
import os
import pickle
import numpy as np

def __grab_patches(images, random_idxs, patch_size=6, tot_patches=1e6, seed=0, scale=0):
    patches = np.zeros((len(random_idxs), images.shape[1], patch_size, patch_size), dtype=images.dtype)
    for i, (im_idx, idx_x, idx_y) in enumerate(random_idxs):
        out_patch = patches[i, :, :, :]
        im = images[im_idx]
        if (scale != 0):
            im = skimage.filters.gaussian(im, sigma=scale)
        grab_patch_from_idx(im, idx_x, idx_y, patch_size, out_patch)
    return patches


def grab_patch_from_idx(im, idx_x, idx_y, patch_size, outpatch): #[idx - 3, idx + 2]
    sidx_x = int(idx_x - patch_size/2.0)
    eidx_x = int(idx_x + patch_size/2.0)
    sidx_y = int(idx_y - patch_size/2.0)
    eidx_y = int(idx_y + patch_size/2.0)
    outpatch[:,:,:] = im[:, sidx_x:eidx_x, sidx_y:eidx_y]
    return outpatch

def grab_patches(images, patch_size=6, tot_patches=5e5, seed=0, max_threads=50, scale=0):
    idxs = chunk_idxs(images.shape[0], max_threads)
    tot_patches = int(tot_patches)
    patches_per_thread = int(tot_patches/max_threads)
    np.random.seed(seed)
    seeds = np.random.choice(int(1e5), len(idxs), replace=False)
    dtype = images.dtype
    tot_patches = int(tot_patches)

    with fs.ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = []
        for i,(sidx, eidx) in enumerate(idxs):
            images.shape[0]
            im_idxs = np.random.choice(images[sidx:eidx, :].shape[0], patches_per_thread)
            idxs_x = np.random.choice(int(images.shape[2]) - patch_size + 1, tot_patches) #[0, 26] 
            idxs_y = np.random.choice(int(images.shape[3]) - patch_size + 1, tot_patches)
            idxs_x += int(np.ceil(patch_size/2.0)) #[3, 29]
            idxs_y += int(np.ceil(patch_size/2.0))
            random_idxs =  list(zip(im_idxs, idxs_x, idxs_y))

            futures.append(executor.submit(__grab_patches, images[sidx:eidx, :],
                                           patch_size=patch_size,
                                           random_idxs=random_idxs,
                                           tot_patches=patches_per_thread,
                                           seed=seeds[i],
                                           scale=scale
                                            ))
        results = np.vstack(list(map(lambda x: x.result(), futures)))
    idxs = np.random.choice(results.shape[0], results.shape[0], replace=False)
    return results[idxs], idxs


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict


def load_cifar10(path):
    train_batches = []
    train_labels = []
    for i in range(1,6):
        cifar_out = unpickle(os.path.join(path, "data_batch_{0}".format(i)))
        train_batches.append(cifar_out[b"data"])
        train_labels.extend(cifar_out[b"labels"])
    X_train= np.vstack(tuple(train_batches)).reshape(-1, 3, 32, 32)
    y_train = np.array(train_labels)
    cifar_out = unpickle(os.path.join(path, "test_batch"))
    X_test = cifar_out[b"data"].reshape(-1, 3, 32, 32)
    y_test = cifar_out[b"labels"]

    return X_train, X_test

def load_cifar100(path):
    train = unpickle(os.path.join(path, "train"))
    test = unpickle(os.path.join(path, "test"))
    return train['data'].reshape(-1, 3, 32, 32), test['data'].reshape(-1, 3, 32, 32)

def load_fashion_mnist(path):
	train = np.load(os.path.join(path, "fashion_mnist_train.npy"))
	test = np.load(os.path.join(path, "fashion_mnist_test.npy"))
	return train, test
def normalize_patches(patches, min_divisor=1e-8, zca_bias=0.001, mean_rgb=np.array([0,0,0])):
    if (patches.dtype == 'uint8'):
        patches = patches.astype('float64')
        patches /= 255.0
    n_patches = patches.shape[0]
    orig_shape = patches.shape
    patches = patches.reshape(patches.shape[0], -1)
    # Zero mean every feature
    patches = patches - np.mean(patches, axis=1)[:,np.newaxis]

    # Normalize
    patch_norms = np.linalg.norm(patches, axis=1)

    # Get rid of really small norms
    patch_norms[np.where(patch_norms < min_divisor)] = 1

    # Make features unit norm
    patches = patches/patch_norms[:,np.newaxis]


    patchesCovMat = 1.0/n_patches * patches.T.dot(patches)

    (E,V) = np.linalg.eig(patchesCovMat)

    E += zca_bias
    sqrt_zca_eigs = np.sqrt(E)
    inv_sqrt_zca_eigs = np.diag(np.power(sqrt_zca_eigs, -1))
    global_ZCA = V.dot(inv_sqrt_zca_eigs).dot(V.T)
    patches_normalized = (patches).dot(global_ZCA).dot(global_ZCA.T)

    return patches_normalized.reshape(orig_shape).astype('float32')

def chunk_idxs(size, chunks):
    chunk_size  = int(np.ceil(size/chunks))
    idxs = list(range(0, size+1, chunk_size))
    if (idxs[-1] != size):
        idxs.append(size)
    return list(zip(idxs[:-1], idxs[1:]))

def chunk_idxs_by_size(size, chunk_size):
    idxs = list(range(0, size+1, chunk_size))
    if (idxs[-1] != size):
        idxs.append(size)
    return list(zip(idxs[:-1], idxs[1:]))
