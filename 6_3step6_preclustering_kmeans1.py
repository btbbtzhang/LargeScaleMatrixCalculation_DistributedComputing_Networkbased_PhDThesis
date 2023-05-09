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
maxiteration = 5
for xo in range(maxiteration):
    comm.Barrier()
    tic = time()
    fin_array = "/global/project/hpcg1553/Yang/ESNA/Shrey/wgcnaTest/datasets/clustering_arrays/final_merged_array_sim.npy"
    array = np.load(fin_array, mmap_mode = 'r')
    #mmap_mode = 'r'
    #print("the info of MERGED array: ", len(array), array[:5], array[187670550000:187670550005])
    print('the time to LOAD DATA is ',time()-tic)
    
    comm.Barrier()
    n = 775569-1
    name1 = 'original'
    name2 = 'sortedIndex'

#tic = time()
#blocksize = 10000
#ncluster = int(min((n+1)/20, (blocksize)**2/(n+1))) # ncluster =128
#initiatedCentres = random.sample(range(0, n+1), ncluster)
#print('which nodes have been initially selected as centres: ',ncluster, len(initiatedCentres), initiatedCentres)

    ncluster = 128
#initiatedCentres = [94728, 205779, 157256, 775286, 317391, 83533, 192696, 316604, 92808, 368704, 756927, 694815, 494888, 566840, 404057, 406743, 527961, 425448, 333694, 754026, 449725, 560145, 661597, 42537, 663243, 269742, 43018, 241813, 537876, 740042, 15254, 598053, 487084, 234604, 55035, 397040, 242986, 92885, 526354, 675589, 449017, 204565, 237651, 215228, 430335, 178352, 320511, 346337, 8719, 112332, 430087, 325608, 340555, 379892, 649144, 212942, 252155, 490719, 183687, 172067, 231805, 324086, 580936, 169924, 461608, 219528, 316396, 667237, 136511, 731786, 753335, 435278, 492495, 614317, 631954, 112160, 418795, 689515, 82691, 266661, 58787, 734954, 732788, 273317, 312738, 329376, 282953, 131594, 87984, 628878, 434269, 473110, 520907, 510485, 71904, 504921, 274905, 575313, 372565, 130853, 294858, 235485, 411333, 586307, 443631, 495340, 618855, 182857, 666469, 290237, 59895, 249729, 667829, 762235, 372544, 438930, 338643, 739148, 370445, 589086, 671586, 4077, 37275, 372262, 747191, 700229, 209765, 512661]
    initiatedCentres = []
    allcentrepoints = np.load('/global/project/hpcg1553/Yang/ESNA/Shrey/wgcnaTest/datasets/clustering_arrays/Kmeans_centrepoints.npy', allow_pickle=True)
    for i in range(size):
        i_coreCentre = allcentrepoints[i]
        i_coreCentre = list(i_coreCentre)
        initiatedCentres = initiatedCentres + i_coreCentre


#idxc = [0,13,26,39,52,65,78,91,104,10,35,12,25,38,51,64,77,90,103,116,22,47,24,37,50,63,76,89,102,115,9,34,59,36,49,62,75,88,101,114,127,21,46,71,48,61,74,87,100,113,126,8,33,58,83,60,73,86,99,112,125,7,20,45,70,95,72,85,98,111,124,6,19,32,57,82,107,84,97,110,123,5,18,31,44,69,94,119,96,109,122,4,17,30,43,56,81,106,108,121,3,16,29,42,55,68,93,118,120,2,15,28,41,54,67,80,105,11,1,14,27,40,53,66,79,92,117,23]

    idxc = [0,12,24,36,48,60,72,84,96,108,120,1,13,25,37,49,61,73,85,97,109,121,2,14,26,38,50,62,74,86,98,110,122,3,15,27,39,51,63,75,87,99,111,123,4,16,28,40,52,64,76,88,100,112,124,5,17,29,41,53,65,77,89,101,113,125,6,18,30,42,54,66,78,90,102,114,126,7,19,31,43,55,67,79,91,103,115,127,8,20,32,44,56,68,80,92,104,116,9,21,33,45,57,69,81,93,105,117,10,22,34,46,58,70,82,94,106,118,11,23,35,47,59,71,83,95,107,119]

    centrepoints = []
    for i in range(len(idxc)):
        idxcc = np.array(idxc)==i
        initiatedCentres = np.array(initiatedCentres)
        #print('initiatedCentres[idxcc]', initiatedCentres[idxcc][0])
        centrepoints.append(initiatedCentres[idxcc][0])

    if rank ==0:
        print('initiatedCentres is', type(initiatedCentres), initiatedCentres )
        print('centrepoints is ', type(centrepoints), centrepoints)

    initiatedCentres = centrepoints
    rowall = np.empty((ncluster, n+1))
    for i in range(ncluster):
        idx = initiatedCentres[i]
        rowi = quickrowcatch(idx, n, array)
        rowall[i,:] = rowi

    print('the shape of matrix containing centre points', rowall.shape, rowall[:,0])
    print('the time to OBTAIN the matrix is ',time()-tic)

    tic = time()
    results_ini_cluster = np.argmax(rowall, axis=0)
    rowsecond = np.array(range(775569))
    #results_ini_cluster = np.vstack((results_ini_cluster,rowsecond))
    print('the time to OBTAIN clustering result is ',time()-tic)

    tic = time()
    clusternumber = np.array(range(ncluster))
    centrepoints = []

    for i in range(rank, ncluster, size):
        idxcluster_i = results_ini_cluster == clusternumber[i]
        cluster_i = rowsecond[idxcluster_i]
        idx1 = cluster_i != initiatedCentres[i]
        cluster_i1 = cluster_i[idx1]
        print('start calculating the iteration for cluster', i, clusternumber[i], 'of ori size', len(cluster_i),'by processor #', rank,'.Imputed size',len(cluster_i1))
        cluster_temp = np.empty((1, len(cluster_i1)))
        #current_cluster = np.empty((len(cluster_i), n+1))
        for j in range(len(cluster_i1)):
            if j%1000==0:
                print('TRACKING the row', j, 'in the cluster of ', i,'by processor #', rank)
            rowj = quickrowcatch(cluster_i1[j], n, array)
            #current_cluster[j,:] = rowj
            rowj = rowj[cluster_i]
            sumrowj = np.nansum(rowj)
            cluster_temp[0,j] = sumrowj
        #current_cluster = current_cluster[:,cluster_i]
        #cluster_temp = np.nansum(current_cluster, axis=1)
        idxmaxcluster_i = np.argmax(cluster_temp)
        centre_i = cluster_i1[idxmaxcluster_i]
        centrepoints.append(centre_i)
        print('the CENTRE point of cluster #', i, ' is ', centre_i)

    print('the time to finish ITERATION #',xo ,' is ',time()-tic, 'by processor # ', rank)

    ALLcentre_i = comm.gather(centrepoints, root = 0)

    if rank ==0:
        print('all centre points after 1st iteration is ', ALLcentre_i)
        os.system('grep MemFree /proc/meminfo')    
        np.save("/global/project/hpcg1553/Yang/ESNA/Shrey/wgcnaTest/datasets/clustering_arrays/Kmeans_centrepoints.npy", ALLcentre_i)

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

#print("FINALLY DONE!!!")

