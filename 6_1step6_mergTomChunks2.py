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

print("Starting step 6: loading arrays ......")
os.system('grep MemTotal /proc/meminfo')
os.system('grep MemFree /proc/meminfo')


######### Loading the SIM matrix

arrays = np.array([], dtype='float16')
ClusteringArrays_dir = "/global/project/hpcg1553/Yang/ESNA/Shrey/wgcnaTest/datasets/clustering_arrays/"

#for i in range(6):
for i in range(6,10):
    start = time()
    array_file = "{}{}forcluster_array.npy".format(ClusteringArrays_dir,i)
    array_current = np.load(array_file)
#    array_current = 1-array_current.astype(np.float32)
    print("Loading clustering array # ", i)
    print("the info of loading current: ", len(array_current), type(array_current),array_current[:2])
    arrays = np.hstack((arrays, array_current))
    del array_current
    print("the length of merged array: ", len(arrays))
    print("It took this much time to merge the clustering arrays: ")
    print(time()-start)
    os.system('grep MemFree /proc/meminfo')

target_path = "/global/project/hpcg1553/Yang/ESNA/Shrey/wgcnaTest/datasets/clustering_arrays/merged_sim_array6to9.npy"
np.save(target_path,arrays)
del arrays
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



