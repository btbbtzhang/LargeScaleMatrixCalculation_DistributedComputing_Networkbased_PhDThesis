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
from mpi4py import MPI
#import multiprocessing as mp
import random
#random.seed(55)

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
tic = time()
fin_array = "/global/project/hpcg1553/Yang/ESNA/Shrey/wgcnaTest/datasets/clustering_arrays/final_merged_array_sim.npy"
array = np.load(fin_array, mmap_mode = 'r')
#mmap_mode = 'r'
#print("the info of MERGED array: ", len(array), array[:5], array[187670550000:187670550005])
print('the time to LOAD DATA is ',time()-tic)

n = 775569-1

#tic = time()
#blocksize = 10000
#ncluster = int(min((n+1)/20, (blocksize)**2/(n+1))) # ncluster =128
#initiatedCentres = random.sample(range(0, n+1), ncluster)
#print('which nodes have been initially selected as centres: ',ncluster, len(initiatedCentres), initiatedCentres)
ncluster = 128
initiatedCentres = [94728,205779,157256,775286,317391,83533,192696,316604,92808,368704,756927,694815,494888,566840,404057,406743,527961,425448,333694,754026,449725,560145,661597,42537,663243,269742,43018,241813,537876,740042,15254,598053,487084,234604,55035,397040,242986,92885,526354,675589,449017,204565,237651,215228,430335,178352,320511,346337,8719,112332,430087,325608,340555,379892,649144,212942,252155,490719,183687,172067,231805,324086,580936,169924,461608,219528,316396,667237,136511,731786,753335,435278,492495,614317,631954,112160,418795,689515,82691,266661,58787,734954,732788,273317,312738,329376,282953,131594,87984,628878,434269,473110,520907,510485,71904,504921,274905,575313,372565,130853,294858,235485,411333,586307,443631,495340,182857,666469,290237,59895,249729,667829,372544,438930,338643,589086,671586,4077,37275,372262,747191,209765,762235,618855,512661,739148,700229,370445]

def getClusters(ncluster, n, initiatedCentres, array):
    print('the length of initiatedCentres is ', len(initiatedCentres))
    rowall = np.empty((ncluster, n+1))
    for i in range(ncluster):
        idx = initiatedCentres[i]
        rowi = quickrowcatch(idx, n, array)
        rowall[i,:] = rowi
    print('the shape of matrix containing centre points', rowall.shape, rowall[:,0])
    results_ini_cluster = np.argmax(rowall, axis=0)
    return results_ini_cluster

tic = time()
if rank ==0:
    results_ini_cluster = getClusters(ncluster, n, initiatedCentres, array)

if rank != 0:
    results_ini_cluster  = np.empty((1, n+1))

results_ini_cluster = comm.Bcast(results_ini_cluster, root=0)
rowsecond = np.array(range(775569))
print('the time to OBTAIN clustering result is ',time()-tic,results_ini_cluster.shape, results_ini_cluster[0:10])

#tic = time()
#results_ini_cluster = np.argmax(rowall, axis=0)
#rowsecond = np.array(range(775569))
##results_ini_cluster = np.vstack((results_ini_cluster,rowsecond))
#print('the time to OBTAIN clustering result is ',time()-tic)

tic = time()
clusternumber = np.array(range(ncluster))
centrepoints = []

for i in range(rank, ncluster, size):
    idxcluster_i = results_ini_cluster == clusternumber[i]
    cluster_i = rowsecond[idxcluster_i]
    print('start calculating the iteration for cluster', i, clusternumber[i],'of size',len(cluster_i), 'by processor #', rank)
    cluster_temp = np.empty((1, len(cluster_i)))
    #current_cluster = np.empty((len(cluster_i), n+1))
    for j in range(len(cluster_i)):
        if j%1000==0:
            print('TRACKING the row', j, 'in the cluster of ', i)
        rowj = quickrowcatch(cluster_i[j], n, array)
        #current_cluster[j,:] = rowj
        rowj = rowj[cluster_i]
        sumrowj = np.nansum(rowj)
        cluster_temp[0,j] = sumrowj
    #current_cluster = current_cluster[:,cluster_i]
    #cluster_temp = np.nansum(current_cluster, axis=1)
    idxmaxcluster_i = np.argmax(cluster_temp)
    centre_i = cluster_i[idxmaxcluster_i]
    centrepoints.append(centre_i)
    #centrepoints = centrepoints + [centre_i]
    print('the CENTRE point of cluster #', i, ' is ', centre_i)

print('the time to finish ONE interation of Kmeans is ',time()-tic)

ALLcentre_i = comm.gather(centrepoints, root = 0)

if rank ==0:
    print('ALL', len(ALLcentre_i), 'centre points after 1st iteration is ', ALLcentre_i)
    os.system('grep MemFree /proc/meminfo')    
    np.save("/global/project/hpcg1553/Yang/ESNA/Shrey/wgcnaTest/datasets/clustering_arrays/Kmeans_preclustering.npy",ALLcentre_i)

#kk_dir = "/global/project/hpcg1553/Yang/ESNA/Shrey/wgcnaTest/datasets/clustering_arrays/KK_table.npy"
#np.save(kk_dir, Kk)
#tic = time()
#Kk = np.load(kk_dir, allow_pickle=True)
#print('the time to load MM table is ',time()-tic)
#print("Obtained Mm talbe, its shape is ", Kk.shape)

#print('MM table checking of row 376994: ', Kk[376994])
#pathx, rowx, namex = getpathname(376994, n)
#print('file info:',pathx,rowx,namex)
#rowXarray = quickreadfromHDF5(pathx, namex, rowx)
#print('checking row 376994 to 115975, 124440, 247047', rowXarray[115975], rowXarray[124440], rowXarray[247047],'max is', max(rowXarray))
#idextest = np.where(rowXarray==rowXarray[115975])
#sortedidx = np.argsort(rowXarray)
#sortedidx = sortedidx[::-1]
#a = sortedidx[0:5]
#print('the idexes', idextest, a, rowXarray[a])

#pathi, rowi, namei = getpathname(367652, n)
#print('file info:',pathi,rowi,namei)
#rowIarray = quickreadfromHDF5(pathi, namei, rowi)

#pathj, rowj, namej = getpathname(371253, n)
#print('file info:',pathj,rowj,namej)
#rowJarray = quickreadfromHDF5(pathj, namej, rowj)

#rowZarray=rowIarray+rowJarray
#rowZarray=rowZarray/2
#maxZarray =np.nanmax(rowZarray)
#idxZ = np.where(rowZarray==maxZarray)
#print('MAX VAL of first new formed array ', maxZarray, idxZ)

#links=computelink(Kk, n)
#os.system('grep MemFree /proc/meminfo')
#del Kk
#links = linkage(array, "average")
#print("the shape of links ", links.shape, links[0:10,:])
#fin_links = "/global/project/hpcg1553/Yang/ESNA/Shrey/wgcnaTest/datasets/clustering_arrays/final_links_sim.npy"
#np.save(fin_links,links)
#os.system('grep MemFree /proc/meminfo')

# clusters = cutreeHybrid(links, array)
#del links
#clusters=clusters["labels"]
#os.system('grep MemFree /proc/meminfo')
#print("the length of cluster labels: ",len(clusters))
#fin_labels = "/global/project/hpcg1553/Yang/ESNA/Shrey/wgcnaTest/datasets/clustering_arrays/cluster_labels_sim.npy"
#np.save(fin_labels,clusters)

print("FINALLY DONE!!! by processor #", rank)
#print("########### Check the memory currently using ##########")
#print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3)
