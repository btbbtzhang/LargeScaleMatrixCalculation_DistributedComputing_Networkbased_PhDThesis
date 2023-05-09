import os
import psutil
import numpy as np
import pandas as pd
from pandas import HDFStore
import gc
import itertools
import h5py
from scipy.sparse import lil_matrix, save_npz
from functools import reduce
from numpy import array
#from multiprocessing import Pool, Process, Manager
gc.collect()

print("Starting reading HDF files in Chunks")
os.system('grep MemTotal /proc/meminfo')
os.system('grep MemFree /proc/meminfo')

filePath = '/global/project/hpcg1553/Yang/ESNA/Shrey/wgcnaTest/datasets/LD06finalGlide_Abs_normed_pyHDFcols.h5'


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
    data = f['/LD06_GlideAbsNormed/table'][:]



print("~~~~~~~~~~~~~~~~~~~~~After array data: ",type(data))
data1=data['Snp1']
print("Numpy array~~~~~~~~~~~~~~~~~~~~~")
print(data1.shape)
print(data1[-10:])
print(data1[1792471230])
print(type(data1[1792471230]))
print("Decoding ~~~~~~~~~~~~~~~~~~~~~")
data1=np.char.decode(data1)
indx1=np.where(data1=='nan')
print("First column check: ")
print(data1[-10:])
data2=data['Snp2']
data2=np.char.decode(data2)
indx2=np.where(data2=='nan')
indxxx=np.unique(np.concatenate((indx1,indx2)))
print("The length of index: ", len(indxxx))
data1=np.delete(data1,indxxx,None)
data2=np.delete(data2,indxxx,None)

gc.collect()
del indx1,indx2
gc.collect()
print("Checking the HDF5 file after removing NAN: ")
data3 = np.delete(data['Abs_TSnp1n2'],indxxx,None)
print('Colmun 3 shape: ',data3.shape)
print('Colmun 2 shape: ',data2.shape)
print('Colmun 1 shape: ',data1.shape)

del indxxx
gc.collect()
print("~~~~~~~ Starting to save NonNAN data ~~~~~~~~")
data = pd.DataFrame(data1)
data['Snp2']=data2
data['Abs_TSnp1n2']=data3
data.columns=["Snp1","Snp2","Abs_TSnp1n2"]
del data1,data2,data3
gc.collect()
print("~~~~~~~~ Pandas! ~~~~~~~ Double checking")
print(data.shape)
print(data.info())
print(data.tail)
store = pd.HDFStore('/global/project/hpcg1553/Yang/ESNA/Shrey/wgcnaTest/datasets/GildeDataNoNA.h5')
store.append('LD06_GlideAbsNormedNoNAN', data, data_columns = data.columns)

os.system('grep MemFree /proc/meminfo')
print("########### Check the memory currently using ##########")
print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3)
print("~~~~~~~~~ Done savings ~~~~~~~~")
