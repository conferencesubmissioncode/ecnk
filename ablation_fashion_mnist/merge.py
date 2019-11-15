import argparse
import time
import numpy as np
import numpy.linalg
import scipy.linalg
import math
import pickle

parser = argparse.ArgumentParser(description='Convolutional Neural Tangent Kernel')
parser.add_argument('-ngpu', default = 40, type = int, help = "Total number of GPU for this task")
parser.add_argument('-d', default = 13, type = int, help = 'Depth')
parser.add_argument('-d_delta', default = 3, type = int, help = 'Depth interval')
parser.add_argument('-d_min', default = 4, type = int, help = 'Min depth')
parser.add_argument('-c', default = 28, type = int, help = 'Crop pixels')
parser.add_argument('-c_delta', default = 4, type = int, help = 'Crop pixels interval')
parser.add_argument('-c_min', default = 0, type = int, help = 'Min crop pixels')
parser.add_argument('-ker_num', default = 2, type = int, help = '2 if Tangent + Convariance Kernel')
parser.add_argument('-bsize', default = 250, type = int, help = 'Block size')
parser.add_argument('-ntrain', default = 60000, type = int, help = "Number of training data")
parser.add_argument('-ntest', default = 10000, type = int, help = "Number of test data")
parser.add_argument('-nvaltrain', default = 50000, type = int, help = "Number of training data in cross validation")
parser.add_argument('-nvaltest', default = 10000, type = int, help = "Number of test data in cross validation")
parser.add_argument('-dir', default = "kernel_full", type = str, help = "Output directory")
alpha = 5e-5
args = parser.parse_args()
N_val_train = args.nvaltrain
N_val_test = args.nvaltest
N_train = args.ntrain
N_test = args.ntest
N_tot = N_train + N_test
N_GPU = args.ngpu
block_size = args.bsize
y_train = pickle.load(open("data/fashion_mnist_train_label")).astype(int).reshape(-1)
y_test = pickle.load(open("data/fashion_mnist_test_label")).astype(int).reshape(-1)
N_class = 10

y_train = y_train[:N_train]
y_test = y_test[:N_train]
Y_train = np.eye(N_class)[y_train]
Y_test = np.eye(N_class)[y_test]

for k in range(0, args.ker_num)[::-1]:
	for d in range(args.d, args.d_min - 1, -args.d_delta):
		for c in range(args.c, args.c_min - 1, -args.c_delta):
			print "k" + str(k) + "d" + str(d) + "c" + str(c) + ":"
			Rx = [] 
			for i in range(N_GPU):
				Rx.append(np.load(args.dir + "/k" + str(k) + "d" + str(d) + "c" + str(c) + "/" + str(i) + ".npy"))
			K = np.zeros((N_tot, N_tot))
			num = 0
			for bi in range(N_tot / block_size):
				for bj in range(bi, N_tot / block_size):
					num = num + 1
					K[bi * block_size : (bi + 1) * block_size, bj * block_size : (bj + 1) * block_size] = Rx[num % N_GPU][num / N_GPU]
					K[bj * block_size : (bj + 1) * block_size, bi * block_size : (bi + 1) * block_size] = Rx[num % N_GPU][num / N_GPU].T
			Rx = [] 
			for i in range(N_GPU):
				Rx.append(np.load(args.dir + "/k" + str(k) + "d" + str(d) + "c" + str(c) + "flip/" + str(i) + ".npy"))
			K_flip = np.zeros((N_tot, N_tot))
			num = 0
			for bi in range(N_tot / block_size):
				for bj in range(bi, N_tot / block_size):
					num = num + 1
					K_flip[bi * block_size : (bi + 1) * block_size, bj * block_size : (bj + 1) * block_size] = Rx[num % N_GPU][num / N_GPU]
					K_flip[bj * block_size : (bj + 1) * block_size, bi * block_size : (bi + 1) * block_size] = Rx[num % N_GPU][num / N_GPU].T
			Rx = None	

			L = np.clip(np.sqrt(np.diag(K)), a_min = 1e-9, a_max = None)
			np.divide(K, L, K)
			np.divide(K, L.reshape(-1, 1), K)
			K[np.diag_indices(N_train)] += alpha
			t = scipy.linalg.solve(K[0:N_train, 0:N_train], Y_train[0:N_train])
			u = K[N_train:N_train + N_test, 0:N_train].dot(t)
			test_acc = 1.0 * np.sum(np.argmax(u, axis = 1) == y_test[0:N_test]) / N_test
			print test_acc
			with open("result.txt", "a+") as f:
				print >> f, "k", k, "d", d, "c", c, test_acc
			K[np.diag_indices(N_train)] -= alpha

			np.divide(K_flip, L, K_flip)
			np.divide(K_flip, L.reshape(-1, 1), K_flip)
			np.add(K, K_flip, K)
			K[np.diag_indices(N_train)] += alpha
			t = scipy.linalg.solve(K[0:N_train, 0:N_train], Y_train[0:N_train])
			u = K[N_train:N_train + N_test, 0:N_train].dot(t)
			test_acc = 1.0 * np.sum(np.argmax(u, axis = 1) == y_test[0:N_test]) / N_test
			print test_acc
			with open("result.txt", "a+") as f:
				print >> f, "k", k, "d", d, "c", c, test_acc
			K[np.diag_indices(N_train)] -= alpha

