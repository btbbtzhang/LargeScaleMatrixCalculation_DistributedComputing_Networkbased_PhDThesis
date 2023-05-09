import os
import psutil
import numpy as np
import pandas as pd
from pandas import HDFStore
import gc
import itertools
import h5py
from scipy.sparse import lil_matrix, save_npz, csr_matrix, coo_matrix
from functools import reduce
from numpy import array
from scipy import sparse
from scipy.io import savemat
from timeit import default_timer as time
#from multiprocessing import Pool, Process, Manager
gc.collect()

print("Starting reading HDF files in Chunks")
os.system('grep MemTotal /proc/meminfo')
os.system('grep MemFree /proc/meminfo')

filePath = '/global/project/hpcg1553/Yang/ESNA/Shrey/wgcnaTest/datasets/GildeLD06DataNoNA.h5'

################ Read data ##############
def traverse_datasets(hdf_file):
    def h5py_dataset_iterator(g, prefix=''):
            for key in g.keys():
                item = g[key]
                path = f'{prefix}/{key}'
                if isinstance(item, h5py.Dataset): # test for dataset
                    yield (path, item)
                elif isinstance(item, h5py.Group): # test for group (go down)        
                    yield from h5py_dataset_iterator(item, path) 
    for path, _ in h5py_dataset_iterator(hdf_file):
        yield path


with h5py.File(filePath, 'r') as f:
    for dset in traverse_datasets(f):
        print('Path:', dset)
        print('Shape:', f[dset].shape)
        print('Data type:', f[dset].dtype)
    data = f['/LD06_GlideAbsNormedNoNAN/table'][:]


print("~~~~~~~~~~~~~~~~~~~~~After array data: ",type(data))
data1=data['Snp1']
print("Numpy array~~~~~~~~~~~~~~~~~~~~~")
print(data1.shape)
print(data1[-10:])

print("Decoding ~~~~~~~~~~~~~~~~~~~~~")
data1=np.char.decode(data1)
print("First column check: ")
print(data1[-10:])
data2=data['Snp2']
data2=np.char.decode(data2)
data3=np.around(data['Abs_TSnp1n2'],decimals=4)
data3=data3.astype('float32')
print(data3[1].dtype)
print('Data3 types: ',data3.dtype)
df_len=len(data1)


def load():
        snp_table = {}
        reverse_lookup = {}
        for row_counter in range(df_len):
            if data1[row_counter] not in snp_table:
                reverse_lookup[len(snp_table)] = data1[row_counter]
                snp_table[data1[row_counter]] = len(snp_table)
        return snp_table, reverse_lookup

os.system('grep MemFree /proc/meminfo')


snp, rev_look = load()
print("It took this much time for it to load row names and data values: ")
os.system('grep MemFree /proc/meminfo')


print('What the type of snp_Talbe and rev_look',type(snp),type(rev_look))
print("GLIDE snp interaction table loadings. Expecting loading rows of matrix: ", len(snp))
print('Check the old rev_look dictionary vector: ', rev_look[1])
rev_look = np.array([list(item.values()) for item in rev_look.values()])
rev_look = np.fromiter(rev_look.values(),dtype=object)
rev_look = np.array(list(rev_look.values()))
print('the row names ',rev_look[0:10])

start = time()
rowNamesOfSIMatrix= '/global/project/hpcg1553/Yang/ESNA/Shrey/wgcnaTest/datasets/RowNamesofSIM.npy'
### Save the SIM matrix row names
np.save(rowNamesOfSIMatrix,rev_look)
print("It took this much time to save the row names: ")
print(time()-start)

sizerow = len(snp)

# lil_matrix is good for the changes of sparse structure of matrix but for operations like slicing, multiplication etc. csr_matrix is better(also it requires less memo) 
#Ref: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html
# csr_matrix is very slow to construct for big size ones, it seems like coo_matrix is the best one that balanced in constructing speed, storage szie, memory using, etc.
#REF: https://datascience.stackexchange.com/questions/31352/understanding-scipy-sparse-matrix-types

#m = csr_matrix((sizerow,sizerow))
#for row_count in range(df_len):
#    m[snp[data1[row_count]], snp[data2[row_count]]] = data3[row_count]
#    if row_count % 50000000 == 0:
#        print('Loaded the numbers of elememts out of 1,792,446,237: ',row_count)

m = lil_matrix((sizerow, sizerow))
for row_count in range(df_len):
    m[snp[data1[row_count]], snp[data2[row_count]]] = data3[row_count]
    if row_count % 50000000 == 0:
        print('Loaded the numbers of elememts out of 1,792,446,237: ',row_count)


print('The shape of Similarity matrix is ',m.shape)
m.setdiag(1)
print('The data type of SIM is: ',m.dtype)
print(m.rows[0:10])
os.system('grep MemFree /proc/meminfo')

######### SAVing the SIM matrix
SIMpath='/global/project/hpcg1553/Yang/ESNA/Shrey/wgcnaTest/datasets/SIMatrix.npz'

def save_sparse_matrix(filename, x):
    xcoo = x.tocoo()
    row = xcoo.row
    col = xcoo.col
    data = xcoo.data
    shape = xcoo.shape
    np.savez(filename, row=row, col=col, data=data, shape=shape)

start = time()
save_sparse_matrix(SIMpath,m)
print("It took this much time to SAVE SIMatrix in COO format: ")
print(time()-start)

os.system('grep MemFree /proc/meminfo')
print("########### Check the memory currently using ##########")
print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3)
