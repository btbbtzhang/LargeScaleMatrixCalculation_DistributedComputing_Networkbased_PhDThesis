import os
import psutil
import numpy as np
#import pandas as pd
from pandas import HDFStore
import gc
import itertools
import h5py
from scipy.sparse import lil_matrix, save_npz, csr_matrix, coo_matrix, vstack
from functools import reduce
from numpy import array
from scipy import sparse
from scipy.io import savemat
import scipy
from timeit import default_timer as time
from multiprocessing import Pool, Process, Manager
from numba import jit, cuda
print(cuda.gpus)
import cupy as cp
import cupy
import cupyx
print('the number of available devices: ',cp.cuda.runtime.getDeviceCount())
gc.collect()

print("Starting step 5: computing TOM matrix by using gpu...")
os.system('grep MemTotal /proc/meminfo')
os.system('grep MemFree /proc/meminfo')

def cpuload_sparse_matrix(filename):
    y = np.load(filename)
    z = sparse.coo_matrix((y['data'], (y['row'], y['col'])), shape=y['shape'],dtype='float16')
    return z


def gpuload_sparse_matrix(filename):
    y = cp.load(filename, mmap_mode='r')
    #y = x[l1:l2]
    z = cupyx.scipy.sparse.coo_matrix((y['data'], (y['row'], y['col'])), shape=y['shape'],dtype='float32')
    return z

######### Loading the SIM matrix
SIMpath='/global/project/hpcg1553/Yang/ESNA/Shrey/wgcnaTest/datasets/SIMatrix.npz'
start = time()
SIMatrix = cpuload_sparse_matrix(SIMpath)

SIMatrix = SIMatrix.tolil()
print("It took this much time to LOAD SIMatrix in LIL format: ")
print(time()-start)
print('The shape of Similarity matrix is ', SIMatrix.shape)
print('The type of SIM is: ', type(SIMatrix))
print(SIMatrix[8,9])

#print(SIMatrix[0:10,0:10])
#os.system('grep MemFree /proc/meminfo')

########print("~~~~~~~~~~~calculating adjacency matrix~~~~~~~~~~")
powerValue=7
AdjMatrix=SIMatrix.power(powerValue,dtype='float16')
del SIMatrix
print("The type of adjacency matrix: ", type(AdjMatrix))
print("The shape of adjacency matrix: ", AdjMatrix.shape)
os.system('grep MemFree /proc/meminfo')

#start = time()
def settdiag(inpu, k):
    diag_mask = inpu.row ==inpu.col
    inpu.data[diag_mask]=k
    return inpu

settdiag(AdjMatrix,0)
#AdjMatrix.setdiag(0)

def save_sparse_matrix(filename, x):
    xcoo = x.tocoo()
    row = xcoo.row
    col = xcoo.col
    data = xcoo.data
    shape = xcoo.shape
    np.savez(filename, row=row, col=col, data=data, shape=shape)

AdjMpath='/global/project/hpcg1553/Yang/ESNA/Shrey/wgcnaTest/datasets/AdjMatrix.npz'
print("Starting to save the adjancency matrix ......")
start = time()
save_sparse_matrix(AdjMpath, AdjMatrix)
print(time()-start)

print("########### Check the memory currently using ##########")
print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3)
