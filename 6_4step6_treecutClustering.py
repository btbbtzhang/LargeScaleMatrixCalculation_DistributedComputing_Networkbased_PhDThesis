import os
import psutil
import numpy as np
import pandas as pd
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
from mpi4py import MPI
#import multiprocessing as mp
import random
random.seed(55)

comm = MPI.COMM_WORLD
rank=comm.Get_rank()
size=comm.Get_size()
print('the size of the total processors are ', size, 'from the process', rank)

if rank == 0:
    print("Starting step 6: loading arrays ......")
    os.system('grep MemTotal /proc/meminfo')
    os.system('grep MemFree /proc/meminfo')


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

def quickreadfromHDF5(path, name, i): #i is the k from the following function, path is also from the next function
    df= h5py.File(path, 'r')
    d = df.get(name)[i]
    return d

def fromRowtoPath(i):
    hdfilepath = "/global/project/hpcg1553/Yang/ESNA/Shrey/wgcnaTest/datasets/TOMatrixRows/"
    TomChunks = "TOMatrixChunk_"
    s=i//3000
    k=i%3000
    Toms_path = "{}{}{}".format(hdfilepath, TomChunks, s)
    return Toms_path, k


def getpathname(a, b):
    if a>b:
        path, row = fromRowtoPath(a-b-1)
        name1 = 'newformed'
        #name2 = 'newformedSortedIdx'
    else:
        path, row = fromRowtoPath(a)
        name1 = 'original'
        #name2 = 'sortedIndex'
    return path, row, name1


os.system('grep MemFree /proc/meminfo')

ncluster = 128

tpath = '/global/project/hpcg1553/Yang/ESNA/Shrey/wgcnaTest/datasets/clustering_arrays/preClusterResults/'
tfile = 'submatrix_fromCluster_'
snpfile = 'SNPsNames_fromCluster_'
finalpath = '/global/project/hpcg1553/Yang/ESNA/Shrey/wgcnaTest/datasets/clustering_arrays/preClusterResults/dynamicTree/'
clusterfile = '_finalClusteredSNPs_'
#i = 1

if rank == 0:
    print("Starting to LOAD the pre-clustering results~~~~~~~~~~~~~~~~")

for i in range(rank, ncluster, size):
    Toms_path = "{}{}{}.npy".format(tpath, tfile, i)
    tom = np.load(Toms_path)
    tom = tom.astype('float32')
    print("what is the length of tom: ", len(tom), 'by the processor', rank)

    disTom = 1-tom
    print("checking if the disTom is good: ", disTom[0:10])

    links = linkage(disTom, "average")
    print('the info of links:', type(links), links.shape)

    clusters = cutreeHybrid(links,disTom, deepSplit =4, minClusterSize = 30, pamRespectsDendro = True)
    #print('The clustering result is ',clusters["labels"][0:99])
    clusteresults = pd.DataFrame(clusters["labels"])
    pd.set_option("display.max_rows", None)
    print('the precluster # ', i, 'by the processor', rank, clusteresults.value_counts())
    
    preclusteredSNPs = "{}{}{}.npy".format(tpath, snpfile, i)
    preclusteredSNPs = np.load(preclusteredSNPs)
    print('the info of preclusteredSNPs: ', preclusteredSNPs.shape, len(preclusteredSNPs))
    print(preclusteredSNPs[0][0:10],type(preclusteredSNPs[0]), len(preclusteredSNPs[0]), len(clusteresults))
    preclusteredSNPs = preclusteredSNPs[0]

    x = list(clusteresults.value_counts().index)
    for j in range(len(x)):
        finalsnps = "{}{}{}{}.npy".format(finalpath, i, clusterfile, j)
        y = np.where(clusteresults==x[j][0])
        idx = y[0]
        print("info of idx: ", type(idx), len(idx), idx[0:10])
        SNPs = preclusteredSNPs[idx]
        np.save(finalsnps, SNPs)
        print('Done Saving dynamicTree Cutting by processor', rank, 'of precluster: ', i, 'for the group ', j)


print('FINALLY finished!!! by the processor ', rank)
