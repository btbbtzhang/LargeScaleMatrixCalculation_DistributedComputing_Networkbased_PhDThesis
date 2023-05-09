from platform import python_version
python_version()

import pandas as pd
import os
import math
from scipy.cluster.hierarchy import linkage
from dynamicTreeCut import cutreeHybrid
from timeit import default_timer as time
from os import listdir 
import unittest
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
#import keras
#from keras.models import *
#from keras.layers import *
#from keras.optimizers import *
#from keras.regularizers import l1, l2
#from keras import backend as K
#pip install matplotlib
#import tensorflow
#import pandas

#load the snp names from the original data matrix (780k x780k)
SNPnames = np.load("/global/project/hpcg1553/Yang/ESNA/Shrey/wgcnaTest/datasets/RowNamesofSIM.npy")

dynamicResults = "/global/project/hpcg1553/Yang/ESNA/Shrey/wgcnaTest/datasets/clustering_arrays/preClusterResults/dynamicTree/"
# There are 914 treecut clusters, 128 precut clusters. 
listfiles = listdir(dynamicResults)
# e.g., 70_finalClusteredSNPs_1.npy. Filename meaning is: 70 presents which precut clusters this file belongs to, 1 means the tag of # 70 precut cluster.
print(len(listfiles), listfiles[2])

## The following code is to save 914 treecutted clustered into one table
prelength = 0
for i in range(len(listfiles)):
    tarfile = os.path.join(dynamicResults, listfiles[i])
    idx = np.load(tarfile)
    setid = "cluster" + str(i)
    currentlength = prelength+idx.shape[0]
    skat_snp[prelength:currentlength, 0] = setid 
    skat_snp[prelength:currentlength, 1]= SNPnames[idx]
    prelength = currentlength
    if i%100 == 0:
        print(i, setid)

savedir = "/global/project/hpcg1553/Yang/ESNA/Shrey/wgcnaTest/datasets/setIDforSKAT_treecut.txt"        
np.save(savedir, skat_snp)
