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

array = np.empty(300753249096, dtype='float16')
os.system('grep MemFree /proc/meminfo')
array1_dir = "/global/project/hpcg1553/Yang/ESNA/Shrey/wgcnaTest/datasets/clustering_arrays/merged_sim_array0to4.npy"
array2_dir = "/global/project/hpcg1553/Yang/ESNA/Shrey/wgcnaTest/datasets/clustering_arrays/merged_sim_array5to9_1.npy"
array1 = array[:187670550000] # this is the length of merged array from array0 to array4.
array1[:] = np.load(array1_dir)
print("the info of array1: ", len(array1), array1[:5])
os.system('grep MemFree /proc/meminfo')
del array1
array2 = array[187670550000:]
array2[:] = np.load(array2_dir)
os.system('grep MemFree /proc/meminfo')
print("the info of array1: ", len(array2), array2[:5])
del array2
print("the info of MERGED array: ", len(array), array[:5], array[187670550000:187670550005])
os.system('grep MemFree /proc/meminfo')

fin_array = "/global/project/hpcg1553/Yang/ESNA/Shrey/wgcnaTest/datasets/clustering_arrays/final_merged_array_sim.npy"
np.save(fin_array,array)
links = linkage(array, "average")
print("the shape of links ", links.shape)
fin_links = "/global/project/hpcg1553/Yang/ESNA/Shrey/wgcnaTest/datasets/clustering_arrays/final_links_sim.npy"
np.save(fin_links,links)
os.system('grep MemFree /proc/meminfo')
clusters = cutreeHybrid(links, SIMarray)
clusters=clusters["labels"]
print("the length of cluster labels: ", len(clusters))
fin_labels = "/global/project/hpcg1553/Yang/ESNA/Shrey/wgcnaTest/datasets/clustering_arrays/cluster_labels_sim.npy"
np.save(fin_labels,clusters)

print("FINALLY DONE!!!")
print("########### Check the memory currently using ##########")
print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3)



