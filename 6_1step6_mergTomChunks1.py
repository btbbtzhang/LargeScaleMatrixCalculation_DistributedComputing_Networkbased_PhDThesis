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
from dynamicTreeCut import cutreeHybrid
from scipy.cluster.hierarchy import linkage
#import multiprocessing as mp
from numba import jit, cuda
print(cuda.gpus)
import cupy as cp
import cupy
import cupyx
from iteration_utilities import deepflatten
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

def converto_1D_SIMarray(TOMatrix):
    SIMarray={}
    colNum = TOMatrix.shape[1]
    print(colNum)
    rowElemOld=0
    for i in range(TOMatrix.shape[0]):
        j=i+1
        rowElem = colNum-j
        rowStart=rowElemOld
        rowEnd = rowStart+rowElem
        for k in range(rowStart,rowEnd):
            SIMarray[k] = TOMatrix[i,k-rowElemOld+j]
#        SIMarray[rowStart:rowEnd] = TOMatrix[i,j:colNum]
        rowElemOld = rowEnd
    return SIMarray

def cphstack(tup):
    arrs = [cupy.atleast_1d(a) for a in tup]
    axis = 1
    if arrs[0].ndim == 1:
        axis = 0
    return concatenate(arrs, axis)

def converthelastom(TOMatrix):
    SIMarray=np.array([])
    rowNum = TOMatrix.shape[0]
    print("the final input matrix shape ", TOMatrix.shape)
    h = 1292*600 # for the second tom chunk "1filenamesomething.npz", we need to - the first chunk 600 rows to start counting, which means for the final chunk #1293 "1292filenamesomething.npz"(python start count from 0), we need to - the previous 1292 chunks * 600 rows (600 rows for each privous chunks).
    rowNum=rowNum-1 # we don't pick up elements from the last row of upper triangle matrix 
    for i in range(rowNum):
        j=i+1+h
        SIMarray1 = TOMatrix[i,:]
        SIMarray1 = SIMarray1.toarray()
        SIMarray1 = SIMarray1[0]
        SIMarray1 = SIMarray1[j:]
        SIMarray = cp.hstack((SIMarray,SIMarray1))
    rowNum=rowNum+1 # to calculate the correct chunk #
    h=h/rowNum
    print("the numbers of elements in TOM chunk # ",h, len(SIMarray))
    del SIMarray1, h
    return SIMarray

def loadtom2array(TOMatrix,h):
    SIMarray=cp.array([],dtype='float16')
    rowNum = TOMatrix.shape[0]
    print("The shape of input matrix is", TOMatrix.shape)
    h=h*rowNum
    for i in range(rowNum):
    # for each Tom chunk, it has 600 rows,775569 cols. For upper triangle matrix
        j=i+1+h         # we take 775569-1 elements for the first row, and -1 element for the next row
        SIMarray1 = TOMatrix[i,:]   # thus, for chunk1 last row, the elements should be 775569-600
        SIMarray1 = SIMarray1.toarray()  # then chunk2 first elements should be 775569-600-1. In this 
        SIMarray1 = SIMarray1[0]    # function j=1+1+h controls the elements selection 
        SIMarray1 = SIMarray1[j:]
        SIMarray = cp.hstack((SIMarray,SIMarray1))
    h=h/rowNum
    print("the numbers of elements in TOM chunk # ",h, len(SIMarray))
    del SIMarray1, h
#    SIMarray = cp.asnumpy(SIMarray)
    return SIMarray


Tom_dir = "/global/project/hpcg1553/Yang/ESNA/Shrey/wgcnaTest/datasets/TOM_Chunks/"
#test_Tom_chunk1 = "/global/project/hpcg1553/Yang/ESNA/Shrey/wgcnaTest/datasets/TOM_Chunks/0part_TOMatrix_chunk_row0_600.npz"
filename="part_TOMatrix_chunk_row"

x=9
l0 = 1000 # 1293 total numbers of TOM chunks
l1 = 1292
tom = np.array([], dtype='float16')
# tom = cp.array([])
for i in range(l0,l1):
    n0 = i*600
    n1 = (i+1)*600
    TOMfile="{}{}{}{}_{}.npz".format(Tom_dir,i,filename,n0,n1)
    TOMchunk = cpuload_sparse_matrix(TOMfile)
    TOMchunk = cupy.sparse.csr_matrix(TOMchunk,dtype='float32')
    os.system('grep MemFree /proc/meminfo')
#    print("the first elements of TOM chunks", TOMchunk[0,1],TOMchunk[0,2],TOMchunk[0,600],TOMchunk[0,601],TOMchunk[0,602])
    tom_current = loadtom2array(TOMchunk,i)
    del TOMchunk
    tom_current = cp.asnumpy(tom_current)
    tom_current = np.array(tom_current,dtype='float16')
    print("this much memo requried for converting; # ", i)
    os.system('grep MemFree /proc/meminfo')
    tom = np.hstack((tom,tom_current))
    print("the type of tom: ", type(tom))
    del tom_current
    os.system('grep MemFree /proc/meminfo')
print("the lenght of converted array from previous 1292 TOM chunks is ", len(tom))
#concatenate with the array from last tom chunk (chunk #1293 with starting file name 1292)
lastom_dir = "/global/project/hpcg1553/Yang/ESNA/Shrey/wgcnaTest/datasets/TOM_Chunks/1292part_TOMatrix_chunk_row775200_775569.npz"
TOMchunk = cpuload_sparse_matrix(lastom_dir)
TOMchunk = cupy.sparse.csr_matrix(TOMchunk,dtype='float32')
os.system('grep MemFree /proc/meminfo')
tom_current = converthelastom(TOMchunk)
del TOMchunk
tom_current = cp.asnumpy(tom_current)
tom_current = np.array(tom_current,dtype='float16')
tom = np.hstack((tom,tom_current))
del tom_current
print("the lenght of FINAL TOM array: ", len(tom))
os.system('grep MemFree /proc/meminfo')
#print("the first elemenst of TOM array [0:3]: ", tom[0:3])
#print("Elements between the joint of chunk1 and chunk2 (465161099:465161102) in TOM array are: ", tom[465161098:465161104])

array_file = "/global/project/hpcg1553/Yang/ESNA/Shrey/wgcnaTest/datasets/clustering_arrays/"
array_path = "{}{}forcluster_array".format(array_file,x)
np.save(array_path,tom)
os.system('grep MemFree /proc/meminfo')

#links = linkage(SIMarray, "average")
#clusters = cutreeHybrid(links, SIMarray)
#cluster_result = "/global/project/hpcg1553/Yang/ESNA/Shrey/wgcnaTest/datasets/cluster_result.npz"
#save_sparse_matrix(cluster_result, clusters)

#Tom_chunk1 = cpuload_sparse_matrix(test_Tom_chunk1)
#print("It took this much time to load TOM matrix: ")
#print(time()-start)
#os.system('grep MemFree /proc/meminfo')
#print("the type of TOM chunk1: ", type(Tom_chunk1))
#print("the size of TOM chunk1: ", Tom_chunk1.shape)
#Tom_chunk1 = cupy.sparse.csr_matrix(Tom_chunk1)
#os.system('grep MemFree /proc/meminfo')
#print("the type of TOM chunk1: ", type(Tom_chunk1))
#print("the value of TOM(0,0) and TOM(0,2): ", Tom_chunk1[0,0],Tom_chunk1[0,2])


#start = time()
#SIMarray = convertSIMarray(Tom_chunk1)
#print("It took this much time to convert TOM chunk: ")
#print(time()-start)
#os.system('grep MemFree /proc/meminfo')
#print("The type of SIM array: ", type(SIMarray))
#print("The values of SIM array: ", SIMarray[0:9])
#print("The size of SIM array: ", len(SIMarray))

#links = linkage(SIMarray, "average")
#clusters = cutreeHybrid(links, SIMarray)

print("########### Check the memory currently using ##########")
print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3)



