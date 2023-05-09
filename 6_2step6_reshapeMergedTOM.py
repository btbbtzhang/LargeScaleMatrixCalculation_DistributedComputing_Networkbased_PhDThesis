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
from iteration_utilities import deepflatten
import pickle
from mpi4py import MPI


print("Starting step 6: loading arrays ......")
os.system('grep MemTotal /proc/meminfo')
os.system('grep MemFree /proc/meminfo')

comm = MPI.COMM_WORLD
rank=comm.Get_rank()
size=comm.Get_size()
print('the size of the total processors are ', size, 'from the process', rank)

# it named min for testing, here we capture the max
def quickrowcatch(i,n,Array):
    l0=int((2*n-i+1)*i/2)
    l1=int(l0+n-i)
    set1=Array[l0:l1]
    set2=np.empty(i,dtype='int16')
    for j in range(len(set2)):
        set2[j]=(i-1)+0.5*(2*n-j-1)*j
    set2=Array[set2]
    set_i=np.hstack((set2,set1))
    del set1, set2
    set_i=np.insert(set_i, i, np.nan, axis=None)
    return set_i

def descendsort(array):
    idxsorted = np.argsort(array)
    idxsorted = idxsorted[::-1]
    return idxsorted

######### Loading the SIM matrix
array = np.empty(300753249096, dtype='float16')
os.system('grep MemFree /proc/meminfo')
fin_array = "/global/project/hpcg1553/Yang/ESNA/Shrey/wgcnaTest/datasets/clustering_arrays/final_merged_array_sim.npy"
array = np.load(fin_array, mmap_mode = 'r')
os.system('grep MemFree /proc/meminfo')
print("the info of MERGED array: ", len(array), array[:5], array[187670550000:187670550005])

#TOMRow_folder = "/global/project/hpcg1553/Yang/ESNA/Shrey/wgcnaTest/datasets/TOMatrixRows/"
#savefileName = "TOMatrixrow_"
n = 775569-1
l0=0
l1=100
#l0 = 200
#l1 = 258
#next l0=100
#for i in range(l0, l1):
#    tic = time()
#    row_i = quickrowcatch(i, n, array)
#    print('the Capturing time for one row of TOMatrix is ', time()-tic)
#    TomRow_path = "{}{}{}.npz".format(TOMRow_folder, savefileName, i)
#    tic = time()
#    np.save(TomRow_path, row_i)
#    print('NP: the Saving time for one row of TOMatrix is ', time()-tic)

hdfilepath = "/global/project/hpcg1553/Yang/ESNA/Shrey/wgcnaTest/datasets/TOMatrixRows/"
TomChunks = "TOMatrixChunk_"
#Toms_path = "{}{}{}".format(hdfilepath, TomChunks, i)

for i in range(l0, l1):
    Toms_path = "{}{}{}".format(hdfilepath, TomChunks, i)
    f = h5py.File(Toms_path, 'w')
    d1 = f.create_dataset('original', (3000,775569), dtype='f16')
    d2 = f.create_dataset('newformed', (3000,775569), dtype='f16')
    for j in range(rank, 3000, size):
        #print(j, 'is written by processor: ', rank)
        if j%1000==0:
            print('Now the row is', j, 'from file:', i ,',which is written into disk by processor:', rank)
        row_j = quickrowcatch(j+i*3000, n, array)
        #print('type of row_j', type(row_j),row_j.shape)
        d1[j,:] = row_j
        #sortedIdx = descendsort(row_j)
        #d2[j,:] = sortedIdx
        del row_j
    f.close()

os.system('grep MemFree /proc/meminfo')

##Toms_path_last = "{}{}{}".format(hdfilepath, TomChunks, 258)
##f = h5py.File(Toms_path_last, 'w')
##d3 = f.create_dataset('original', (1569,775569), dtype='f16')
##d4 = f.create_dataset('newformed', (1569,775569), dtype='f16')
##for j in range(rank, 1569, size):
##    row_j = quickrowcatch(j+774000, n, array)
##    d3[j,:] = row_j
##    del row_j
##f.close()


#for i in range(l0, l1):
#    Toms_path = "{}{}{}".format(hdfilepath, TomChunks, i)
#    with h5py.File(Toms_path, 'w', driver='mpio',comm=MPI.COMM_WORLD) as f:
#        d1 = f.create_dataset('original', (1000,775569), dtype='f16')
#        d2 = f.create_dataset('sortedIndex', (1000,775569), dtype='i')
#        for j in range(rank, 1000, size):
#            print(j, 'is written by processor: ', rank)
#            row_j = quickrowcatch(j, n, array)
#            print('type of row_i', type(row_i),row_i.shape)
#            d1[j,:] = row_j
#            sortedIdx = descendsort(row_j)
#            d2[j,:] = sortedIdx
#            del row_j, sortedIdx


#testpath = "/global/project/hpcg1553/Yang/ESNA/Shrey/wgcnaTest/datasets/TOMatrixRows/TOMatrixChunk_2"
#with h5py.File(testpath,'a') as f:
#    d1 = f['original']
#    d2 = f['sortedIndex']
#    print('the shape of d1 and d2 is:', type(d1), d1.shape, d2.shape)


#testds = "/global/project/hpcg1553/Yang/ESNA/Shrey/wgcnaTest/datasets/TOMatrixRows/Test_hdf5_TOMrows.h5"
#with h5py.File(testds,'w') as f:
#    dset = f.create_dataset('testhdf5', (10,775569), dtype='f16')

#os.system('grep MemFree /proc/meminfo')

#tic = time()
#with h5py.File(testds,'a') as f:
#    dset = f['testhdf5']
#    print('The type of dset: ', type(dset))
#    for i in range(l0, l1):
#        row_i = quickrowcatch(i, n, array)
#        print('type of row_i', type(row_i),row_i.shape)
#        dset[i,:] = row_i
#print('HDF5: the Saving time for 10 rows of TOMatrix is ', time()-tic)

#os.system('grep MemFree /proc/meminfo')

#tic = time()
#with h5py.File(testds,'r') as f:
#    d1= f['testhdf5'][6]
    
#print('HDF5: the Reading time for one row of TOMatrix is ', time()-tic)
#print(type(d1))
#print(d1.shape)

#os.system('grep MemFree /proc/meminfo')

#hf = h5py.File(df, 'w')
#hf.create_dataset('data0', data=row_i)
#hf.close()
#print('HDF5: the Saving time for one row of TOMatrix is ', time()-tic)

#pikl = "/global/project/hpcg1553/Yang/ESNA/Shrey/wgcnaTest/datasets/TOMatrixRows/TOMrow0.pickle"
#tic = time()
#with open(pikl, 'wb') as handle:
#    pickle.dump(row_i, handle, protocol=pickle.HIGHEST_PROTOCOL)
#print('PICKLE: the Saving time for one row of TOMatrix is ', time()-tic)

print("Finishing storing!!! ")
print("########### Check the memory currently using ##########")
print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3)
