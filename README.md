# ParallelComputing_GPUimplementation (Only partial code here, since paper is recently submitted)

## Aim:
To create a novel model that can address large genetic interactions problem in genome wide level.

## How to reach:
Cutting machine-learning algorithms by distributed computing, GPU acceleration, static fixed memory size, suitable data structures, etc.

## WorkFlow:  
Preprocessing big genetic data: step1, step2  

Converting table data into square similarity matrix (780k by 780k): step3  

Denoising the big matrix by using power exponential method: step4  

Distributing the Topological Matrix Calculation into 5 different nodes: step5  

Merging the results from different nodes: step6_1, step6_2  

Opitimizing/rewriting the clustering method, Kmeans by using homogenerous computing on the same node to cluster the merged result: step6_3  

Futher clustering the pre-clustered results: step6_4  

Interpreting the clustered results: step7  

## Result:
The first tool that can successfully handle a big matrix calculations for genome-wide genotype interactions data (780kx 780k matrix) in bioinformatics field. Increased the efficiency from 3 years to 15 days. Saved 40%~ 50% of memory consumptions. Successfully found the cross-validated biological findings in the target signal pathways, which proved our model accurate and precise.  

<img width="612" alt="Screen Shot 2023-05-09 at 7 41 05 PM" src="https://github.com/btbbtzhang/ParallelComputing_GPUimplementation/assets/34163897/218e5a3e-7d53-4e35-acc4-3475e91014b4">



Landscape Pipeline of this project by using snakemake:  
![ESNApipeline](https://github.com/btbbtzhang/ParallelComputing_GPUimplementation/assets/34163897/3a577460-fcba-45d9-be41-1cddc4c0e040)
