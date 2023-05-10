import os
import psutil
import numpy as np
#import pandas as pd
from pandas import HDFStore
import gc
import math
import itertools
import h5py
from scipy.sparse import lil_matrix, save_npz, csr_matrix, coo_matrix, vstack
from functools import reduce
from numpy import array
from scipy import sparse
from scipy.io import savemat
import scipy
from timeit import default_timer as time
import multiprocessing as mp
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


######### Loading the SIM matrix

def cpuload_sparse_matrix(filename):
    y = np.load(filename)
    z = sparse.coo_matrix((y['data'], (y['row'], y['col'])), shape=y['shape'],dtype='float16')
    return z

def gpuload_sparse_matrix(filename):
    y = cp.load(filename, mmap_mode='r')
    #y = x[l1:l2]
    z = cupyx.scipy.sparse.coo_matrix((y['data'], (y['row'], y['col'])), shape=y['shape'],dtype='float32')
    return z

def cpdivi(self, other):
    z=self.todense()/other.todense()
    return cupy.sparse.csr_matrix(z)


def settdiag(inpu, k):
    diag_mask = inpu.row ==inpu.col
    inpu.data[diag_mask]=k
    return inpu

def save_sparse_matrix(filename, x):
    xcoo = x.tocoo()
    row = xcoo.row
    col = xcoo.col
    data = xcoo.data
    shape = xcoo.shape
    np.savez(filename, row=row, col=col, data=data, shape=shape)

def gpusave_sparse_matrix(filename, x):
    xcoo = x.tocoo()
    row = xcoo.row
    col = xcoo.col
    data = xcoo.data
    shape = xcoo.shape
    cp.savez(filename, row=row, col=col, data=data, shape=shape)

@cuda.jit
def cuda_matmul(A, B, C):
    """Perform matrix multiplication of C = A * B
    """
    row, col = cuda.grid(2)
    if row < C.shape[0] and col < C.shape[1]:
        tmp = 0.
        for k in range(A.shape[1]):
            tmp += A[row, k] * B[k, col]
        C[row, col] = tmp

@cuda.jit
def matrices2_adding(A,B):
    return (A+B)

@cuda.jit
def fast_matmul(A, B, C):
    # Define an array in the shared memory matrix: sa and sb
    # The size and type of the arrays must be known at compile time
    sa = cuda.shared.array(shape=(TPB, TPB), dtype=float32) # TPB: threads per block
    sb = cuda.shared.array(shape=(TPB, TPB), dtype=float32)

    x, y = cuda.grid(2)

    idx = cuda.threadIdx.x
    idy = cuda.threadIdx.y
    bpg = cuda.gridDim.x    # blocks per grid

    # Each thread computes one element in the result matrix.
    # The dot product is chunked into dot products of TPB-long vectors.
    tmp = float32(0.)
    for i in range(bpg):
        # Preload data into shared memory. Reordered the sense of x and y (and usage of tx and ty) to fix a performance issue, it works less efficiently if we do sa[idx, idy]
        sa[idy, idx] = 0
        sb[idy, idx] = 0
        if y < A.shape[0] and (idx+i*TPB) < A.shape[1]:
          sa[idy, idx] = A[y, idx + i * TPB]
        if x < B.shape[1] and (idy+i*TPB) < B.shape[0]:
          sb[idy, idx] = B[idy + i * TPB, x]

        # Wait until all threads finish preloading
        cuda.syncthreads()

        # Computes partial product on the shared memory
        for j in range(TPB):
            tmp += sa[idy, j] * sb[j, idx]

        # Wait until all threads finish computing
        cuda.syncthreads()
    if y < C.shape[0] and x < C.shape[1]:
        C[y, x] = tmp

def Tom_calc(A,L,ki,kj,n0,n1,Yi):
    os.system('grep MemFree /proc/meminfo')
    A_tom1 = L + A # size 100k by AdjMarix.shape[1]
    del L
    print("Got A_tom1 = L + A~~~")
    os.system('grep MemFree /proc/meminfo')
    MINK=lil_matrix((n1-n0,A_tom1.shape[1]),dtype='float32')
    for j in range(n1-n0):
        MINK[j,:]=kj.minimum(ki[j,0])

    del ki,kj
    print('the info of MINK chunk1: ',MINK.shape)
#    print("check the MINK data type: ", type(MINK))
#    print("check the MINK: ", MINK[0:2,0:2])
    print("MINK obtained here ~~~~")
    os.system('grep MemFree /proc/meminfo')

    MINK=MINK.tocsr()
    MINK=cupy.sparse.csr_matrix(MINK)
#    Yi=cupy.sparse.csr_matrix(cp.ones((600,775569)),dtype='float32')
    A_tom2 = MINK+Yi-A
#    del Yi
    print("check the TOM_part2 data type: ", type(A_tom2))
    print("check the TOM_part2(1,3) value: ", A_tom2.A[1,3])
    print("Got A_tom2 = MINK+Yi-A here~~~")
    os.system('grep MemFree /proc/meminfo')

    del A, MINK
    A_tom_chunk1 = cpdivi(A_tom1,A_tom2)
    cp.cuda.Stream.null.synchronize()
    del A_tom1, A_tom2
    print("the info of TOM chunk1: ", A_tom_chunk1.shape)
    print("the type of TOM chunk1: ", type(A_tom_chunk1))
    print("Got A_tom_chunk1 = A_tom1/ A_tom2 here~~~")
    gc.collect()
    A_tom_chunk1=A_tom_chunk1.tocoo()
    settdiag(A_tom_chunk1,1)
    print("the type of TOM chunk1: ", type(A_tom_chunk1))
    return A_tom_chunk1


AdjMpath='/global/project/hpcg1553/Yang/ESNA/Shrey/wgcnaTest/datasets/AdjMatrix.npz'

"""
start = time()
AdjMatrix = cpuload_sparse_matrix(AdjMpath)
print("Starting to load the adjancency matrix ......")
start = time()
print("It took this much time to load adjancency matrix: ")
print(time()-start)
os.system('grep MemFree /proc/meminfo')
"""

#TPB must be an integer between 1 and 32
TPB = 3 # need to find a number that match with the matrix dimension
threadsperblock = (TPB, TPB)
# make sure the grid is big enough to cover the entire input matrix
grid_y_max = max(A.shape[0], AdjMatrix.shape[0])
grid_x_max = max(A.shape[1], AdjMatrix.shape[1])
blockspergrid_x = math.ceil(grid_x_max / threadsperblock[0]) # we have 775569 data points, so 775569/3=258523
blockspergrid_y = math.ceil(grid_y_max / threadsperblock[1])
blockspergrid = (blockspergrid_x, blockspergrid_y)

Yi=cupy.sparse.csr_matrix(cp.ones((369,775569)),dtype='float32') # 369=n1-n0
Savdirectorypath="/global/project/hpcg1553/Yang/ESNA/Shrey/wgcnaTest/datasets/TOM_Chunks/"
Savfilename="part_TOMatrix_chunk_row"


i=1292
# CUPY version 6.5 is too old !!! CAC's limitation!!!
n0=i*600
n1=775569
print("Part: ",i,"__row: ", n0,"to",n1)
print("Starting to load the adjancency matrix ......")
AdjMatrix = cpuload_sparse_matrix(AdjMpath)
AdjMatrix=AdjMatrix.tocsr()
A=AdjMatrix[n0:n1,:]
# kj is (1, 775569) and ki is (600, 1)
kj = lil_matrix(AdjMatrix.sum(axis=0))
ki = A.sum(axis=1)
AdjMatrix=cupy.sparse.csr_matrix(AdjMatrix)
A=cupy.sparse.csr_matrix(A)
print("the data shape of A: ", A.shape)
#cuda_matmul[775569, 600](A, AdjMatrix, L) this cuda function exceed the max griddimension
fast_matmul[blockspergrid, threadsperblock](A, AdjMatrix, L)
#L = A * AdjMatrix
print("L = A * AdjMatrix is obtained")
del AdjMatrix

# Compute TOM!!!!!!!!
A_tom_chunk1=Tom_calc(A,L,ki,kj,n0,n1,Yi)
del Yi
print('The TOM matrix is calculated!')
os.system('grep MemFree /proc/meminfo')
#    A_tom_chunk1=A_tom_chunk1.tocsr()
#    print('Some values of TOM matrix chunk1(3,0)',A_tom_chunk1[3,0])
#    print(A_tom_chunk1[3,3])
#    os.system('grep MemFree /proc/meminfo')
TOMpath="{}{}{}{}_{}.npz".format(Savdirectorypath,i,Savfilename,n0,n1)
print("Starting to save the TOM matrix chunks (every 600 rows), from row ", n0,"to ",n1)
start = time()
save_sparse_matrix(TOMpath, A_tom_chunk1)
print("It took this much time to save TOM chunk: ")
print(time()-start)
del A_tom_chunk1

print("########### Check the memory currently using ##########")
print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3)

os.system('grep MemFree /proc/meminfo')


