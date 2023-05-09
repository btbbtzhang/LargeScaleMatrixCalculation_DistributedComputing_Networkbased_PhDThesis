print("starting") #starting loading normalization 
import os
import numpy as np
import pandas as pd
# from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MaxAbsScaler
import json
import gc
from pandas import HDFStore

gc.collect()


prepro_output_dir = "/global/project/hpcg1553/CHILD_glide/genome_wide_LD_06/1preprocessing/outputs/"
glide_output_dir = "/global/project/hpcg1553/CHILD_glide/genome_wide_LD_06/2glide/outputs/"

snplist_path = os.path.join(glide_output_dir, "CHILD_maf005_Caucassian_prunedInLD06_T3glideout_snplist")
GLIDEoutput_path = os.path.join(glide_output_dir, "CHILD_maf005_Caucassian_prunedInLD06_T3glideout_wSnpIDandPvals_clean")


#################   ---  Loading data  ---   ####################
os.system('grep MemTotal /proc/meminfo')
os.system('grep MemFree /proc/meminfo')

#load snp interaction glide result
glide_output = pd.read_table(GLIDEoutput_path, delim_whitespace=True, names=['Snp1','Snp2','TSnp1n2'])
print("glide_output dim = ", glide_output.shape)
print("Tscore min = ", glide_output['TSnp1n2'].min())
print("Tscore max = ", glide_output['TSnp1n2'].max())
os.system('grep MemFree /proc/meminfo')


#################   ---  Normalizing data  ---   ####################
# create an abs_scaler object by using sklearn library.
abs_scaler = MaxAbsScaler()
transformer = glide_output.copy()
print("Orginal glide results type")
print(type(glide_output))
print("#######################################################################")
print("Check the copy of glide: transformer")
print(transformer.head())
print(transformer.shape)

# calculate the maximum absolute value for scaling the data using the fit method
transformer["Abs_TSnp1n2"] = abs_scaler.fit_transform(transformer[["TSnp1n2"]])
transformer["Abs_TSnp1n2"] = np.absolute(transformer[["Abs_TSnp1n2"]])

print("Check the normalized glide results:")
print(transformer.head())
#transformer['TSnp1n2'] = transformer1.to_numpy()
transformer = pd.DataFrame(transformer, columns=transformer.columns)
print("Double check the transformed data: transformer")
print(type(transformer))
print(transformer.shape)
print(transformer.head())
print(transformer.tail())
print("Normed glide result dim = ", transformer.shape)
print("Tscore normed min = ", transformer['Abs_TSnp1n2'].min())
print("Tscore normed max = ", transformer['Abs_TSnp1n2'].max())

#################### --- store the results --- ###########################
del glide_output
gc.collect()
os.system('grep MemFree /proc/meminfo')
store = pd.HDFStore('/global/project/hpcg1553/Yang/ESNA/Shrey/wgcnaTest/rdata/LD06finalGlide_Abs_normed_pyHDFcols.h5')
store.append('LD06_GlideAbsNormed', transformer, data_columns = transformer.columns)
os.system('grep MemFree /proc/meminfo')
