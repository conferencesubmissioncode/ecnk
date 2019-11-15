import argparse
import time
import config
import numpy as np
import pickle
import cupy as cp
import json


parser = argparse.ArgumentParser(description='Convolutional Neural Tangent Kernel')
parser.add_argument('-l', type = str, required = True, help = "Kernel filename")
parser.add_argument('-ngpu', default = 10, type = int, help = "Total number of GPU for this task")
parser.add_argument('-lgpu', default = 10, type = int, help = "Number of GPU on this server")
parser.add_argument('-gpu', default = 0, type = int, help = 'GPU ID')
parser.add_argument('-d', default = 19, type = int, help = 'Depth')
parser.add_argument('-d_delta', default = 1000, type = int, help = 'Depth interval')
parser.add_argument('-d_min', default = 0, type = int, help = 'Min depth')
parser.add_argument('-c', default = 16, type = int, help = 'Crop pixels')
parser.add_argument('-c_delta', default = 4, type = int, help = 'Crop pixels interval')
parser.add_argument('-c_min', default = 0, type = int, help = 'Min crop pixels')
parser.add_argument('-ker_num', default = 1, type = int, help = '2 if Tangent + Convariance Kernel')
parser.add_argument('-bsize', default = 250, type = int, help = 'Block size')
parser.add_argument('-pixel', default = 34, type = int, help = 'Pixels')
parser.add_argument('-bias', default = 1.0, type = float, help = "CNTK bias")
parser.add_argument('-f', default = "no", type = str, help = "Horizontal flip")
parser.add_argument('-pool', default = "LAP", type = str, help = "Pooling type")
parser.add_argument('-ntrain', default = 50000, type = int, help = "Number of training data")
parser.add_argument('-ntest', default = 10000, type = int, help = "Number of test data")
parser.add_argument('-timing', default = "no", type = str, help = "Printing running time")
parser.add_argument('-printing', default = "no", type = str, help = "Printing kernel values")
parser.add_argument('-test', default = "no", type = str, help = "Testing mode")
parser.add_argument('-debug', default = "no", type = str, help = "Debug mode")
parser.add_argument('-estimate', default = "no", type = str, help = "Time estimating mode")
parser.add_argument('-dir', default = "kernel", type = str, help = "Output directory")
parser.add_argument('-feature', default = "coatesng", type = str, help = "Feature extractor")
parser.add_argument('-fargs', default = '''{
"dataset":"cifar10",
"patch_size":5,
"num_filters":2048,
"bias":1.0
}''', type = str, help = "Feature extractor arguments")
args = parser.parse_args()


if args.test == "yes":
	args.ntrain = 5000
	args.ntest = 2000
	args.timing = "no"
	args.printing = "no"

if args.debug == "yes":
	args.ngpu = 1
	args.lgpu = 1
	args.gpu = 0
	args.ntrain = 5
	args.ntest = 0
	args.bsize = 5
	args.timing = "yes"
	args.printing = "yes"
	
if args.estimate == "yes":
	args.ngpu = 1
	args.lgpu = 1
	args.gpu = 0
	args.ntrain = args.bsize
	args.ntest = args.bsize
	args.timing = "yes"
	args.printing = "no"

cp.cuda.Device(args.gpu % args.lgpu).use()

config.d = args.d
config.d_delta = args.d_delta
config.d_min = args.d_min
config.c = args.c
config.c_delta = args.c_delta
config.c_min = args.c_min
config.bias = args.bias
config.pool_type = args.pool
config.ker_num = args.ker_num
config.d_num = (args.d - args.d_min) / args.d_delta + 1
config.c_num = (args.c - args.c_min) / args.c_delta + 1
config.pixel = args.pixel
config.flip = (args.f == "yes")
config.fargs = json.loads(args.fargs)

N_GPU = args.ngpu
N_train = args.ntrain
N_test = args.ntest
N_tot = N_train + N_test
block_size = args.bsize
pixel = args.pixel
import code
feature = __import__(args.feature)
pack = __import__(args.l)

print args

assert(config.c <= config.pixel)
assert(N_train % block_size == 0)
assert(N_test % block_size == 0)

N = N_tot / block_size * (N_tot / block_size + 1) / 2
K = np.zeros((config.ker_num, config.d_num, config.c_num, N / N_GPU + 1, block_size, block_size), dtype = cp.float32)
K_flip = None
if config.flip:
    K_flip = np.zeros((config.ker_num, config.d_num, config.c_num, N / N_GPU + 1, block_size, block_size), dtype = cp.float32)

if config.flip:
    print "Total memory:", K.size * args.lgpu * 8.0 / 1024 / 1024 / 1024, "GB"
else:
    print "Total memory:", K.size * args.lgpu * 4.0 / 1024 / 1024 / 1024, "GB"

def get_data(b, flip):
    if (b + 1) * block_size <= N_train:
        return feature.featurize(b * block_size, (b + 1) * block_size, train = True, flip = flip)
    else:
        return feature.featurize(b * block_size - N_train, (b + 1) * block_size - N_train, train = False, flip = flip)
        
D = cp.zeros((N_tot, pixel, pixel), dtype = cp.float32)
D_flip = None

for b in range(N_tot / block_size):
    X = get_data(b, flip = False)
    D[b * block_size : (b + 1) * block_size] = cp.sum(X * X, axis = 1).reshape(-1, pixel, pixel)
    X = None
    
    
if config.flip:
    D_flip = cp.zeros((N_tot, pixel, pixel), dtype = cp.float32)
    for b in range(N_tot / block_size):
        X_flip = get_data(b, flip = True)
        D_flip[b * block_size : (b + 1) * block_size] = cp.sum(X_flip * X_flip, axis = 1).reshape(-1, pixel, pixel)
        X_flip = None


starting_time = time.time()

num = 0
for bi in range(N_tot / block_size):
    X_i = None
    X_i = get_data(bi, flip = False)
    if config.flip:
        X_flip_i = None
        X_flip_i = get_data(bi, flip = True)
    for bj in range(bi, N_tot / block_size): 
        num = num + 1
        print "GPU", args.gpu, 100.0 * num / N, "%"
        if num % N_GPU == args.gpu:
            X_j = None
            X_j = get_data(bj, flip = False)            
            for i in range(block_size):
                for j in range(block_size):
                    K[:, :, :, num / N_GPU, i, j] = pack.xz(X_i[i], X_j[j], D[bi * block_size + i], D[bj * block_size + j]).get()
                    if config.flip:
                        K_flip[:, :, :, num / N_GPU, i, j] = pack.xz(X_flip_i[i], X_j[j], D_flip[bi * block_size + i], D[bj * block_size + j]).get()

if args.timing == "yes":
	t = time.time() - starting_time    
	print "Time: ", t
	print "Estimate GPU hours: ", t * (60000.0 / block_size * (60000.0 / block_size + 1) / 2) / N / 3600.0

if args.printing == "yes":
	for k in range(config.ker_num):
		for i in range(config.d_num):
			for j in range(config.c_num):
				print "Kernel values k = " + str(k) + " d = " + str(config.d - i * config.d_delta) + " c = " + str(config.c - j * config.c_delta) + ":", K[k][i][j][1:]

import os
if not os.path.exists(args.dir):
	try:
		os.makedirs(args.dir)
	except OSError:
		pass

for k in range(config.ker_num):
	for i in range(config.d_num):
		for j in range(config.c_num):
			dir = args.dir + "/k" + str(k) + "d" + str(config.d - i * config.d_delta) + "c" + str(config.c - j * config.c_delta)
			if not os.path.exists(dir):
				try:
					os.makedirs(dir)
				except OSError:
					pass
			np.save(dir + "/" + str(args.gpu), K[k][i][j])
			if config.flip:
				dir = args.dir + "/k" + str(k) + "d" + str(config.d - i * config.d_delta) + "c" + str(config.c - j * config.c_delta) + "flip"
				if not os.path.exists(dir):
					try:
						os.makedirs(dir)
					except OSError:
						pass
				np.save(dir + "/" + str(args.gpu), K_flip[k][i][j])
