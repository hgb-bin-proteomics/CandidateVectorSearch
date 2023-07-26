# CandidateVectorSearch

Searching for candidates using sparse matrix + [sparse] vector/matrix multiplication.

Implements the following methods across two DLLs:
- VectorSearch.dll:
  - findTopCandidates: sparse matrix - sparse vector multiplication using Eigen
  - findTopCandidatesBatched: sparse matrix - sparse matrix multiplication using Eigen
- VectorSearchCUDA.dll:
  - findTopCandidates: sparse matrix - dense vector multiplication using CUDA ([SpMV](https://docs.nvidia.com/cuda/cusparse/index.html#cusparsespmv))
  - findTopCandidatesBatched: sparse matrix - sparse matrix multiplication using CUDA ([SpGEMM](https://docs.nvidia.com/cuda/cusparse/index.html#cusparsespgemm))
  - findTopCandidatesBatched2: sparse matrix - dense matrix multiplication using CUDA ([SpMM](https://docs.nvidia.com/cuda/cusparse/index.html#cusparsespmm))
 
VectorSearch.dll implements functions that run on the CPU, while VectorSearchCUDA.dll implements functions that run on a NVIDIA GPU using CUDA (version 12.2.0_536.25_windows).

Which functions should be used depends on the problem size and the available hardware. A recommendation is depicted at the figure at the bottom.

## Benchmarks

tbd

## Functions

![Method_Decision_Tree](sparse_alg_decision_tree.png)
