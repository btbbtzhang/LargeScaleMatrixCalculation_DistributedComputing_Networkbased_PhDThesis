# ParallelComputing_GPUimplementation

# WorkFlow:  

Preprocessing big genetic data: step1, step2  

Converting table data into square similarity matrix (780k by 780k): step3  

Denoising by using power exponential method: step4  

Distributing the Topological Matrix Calculation into 5 different nodes: step5  

Merging the results from different nodes: step6_1, step6_2  

Opitimizing/rewriting the clustering method, Kmeans by using homogenerous computing on the same node to cluster the merged result: step6_3  

Futher clustering the pre-clustered results: step6_4  

Interpreting the clustered results: step7  

