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
import rpy2
print(rpy2.__version__)
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
import rpy2.robjects.packages as rpackages
from rpy2.robjects.vectors import StrVector
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
gc.collect()

print("Starting step 4 loading SIM and compute soft threshold power...")
os.system('grep MemTotal /proc/meminfo')
os.system('grep MemFree /proc/meminfo')



######### Loading the SIM matrix
SIMpath='/global/project/hpcg1553/Yang/ESNA/Shrey/wgcnaTest/datasets/SIMatrix.npz'

def load_sparse_matrix(filename):
    y = np.load(filename)
    z = sparse.coo_matrix((y['data'], (y['row'], y['col'])), shape=y['shape'],dtype='float16')
    return z

start = time()
SIMatrix = load_sparse_matrix(SIMpath).tolil()
print("It took this much time to LOAD SIMatrix in LIL format: ")
print(time()-start)
print('The shape of Similarity matrix is ', SIMatrix.shape)
print('The data type of SIM is: ', SIMatrix.dtype)
print(SIMatrix[0:16,0:16])
os.system('grep MemFree /proc/meminfo')



start = time()
snpNames = np.load('/global/project/hpcg1553/Yang/ESNA/Shrey/wgcnaTest/datasets/RowNamesofSIM.npy')
print("It took this much time to load the snp names of SIMatrix: ")
print(time()-start)
print(len(snpNames))


print("~~~~~~~~~~~get the power vector~~~~~~~~~~")
base=importr('base')
powers=np.array(list(range(1,11))+list(range(12,32,2)))
powers1=np.concatenate((np.array(list([0])),powers[0:len(powers)-1]))
powerstep=powers-powers1
uniq=np.unique(powers-powers1)
SIMatrixPower2=SIMatrix.power(2)# uniq[0]=1 uniq[1]=2; Jump=1,2
corxPrev=1
x=1
#corxPrev=sparse.lil_matrix(np.ones((775569,775569)),dtype='int8')
datk=lil_matrix((775569,len(powers)))
os.system('grep MemFree /proc/meminfo')

# Customize function for calculating the soft threshold value, where corxCur is similarity matrix here. 
#Comparing to original funciton, corxCur is the calculated PearsonCorrelation there
for i in range(len(powers)):
    if powerstep[i]==1 and x==1:
        corxCur = SIMatrix
        datk[:,i] = corxCur.sum(axis=1)-1
        corxPrev=corxCur
        x+=1
        print(i)
        continue
    if powerstep[i]==1 and corxPrev.shape[0]!=1:
        corxCur = corxPrev.multiply(SIMatrix)
        datk[:,i] = corxCur.sum(axis=1)-1
        corxPrev=corxCur
        print(i)
    if powerstep[i]==2:
        corxCur = corxPrev.multiply(SIMatrixPower2)
        datk[:,i] = corxCur.sum(axis=1)-1
        corxPrev=corxCur
        print(i)

os.system('grep MemFree /proc/meminfo')
print("Release memory~~~~~~~~~~~~")
del powerstep, powers1, uniq, SIMatrixPower2,corxPrev,corxCur
gc.collect()
print("Checking if memory is released~~~~~~~~")
os.system('grep MemFree /proc/meminfo')

# via Python calling R package to use scaleFreeFitIndex function (do not need to customize this function here)
print("Calculation of fitting statistics for evaluating scale free topology fit")
wgcna = importr('WGCNA')
sft = np.zeros((3,len(powers)))
for j in range(len(powers)):
    SFT1 = wgcna.scaleFreeFitIndex(k = datk[:,j].A, nBreaks = 10, removeFirst = False)
    sft[:,j]=list(SFT1[0])
    print(SFT1[0])
    
    
# testing
#sft = wgcna.pickSoftThreshold(SIMatrix.A, powerVector = powers, verbose = 5, networkType = "signed")
print(sft)

os.system('grep MemFree /proc/meminfo')
print("########### Check the memory currently using ##########")
print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3)
