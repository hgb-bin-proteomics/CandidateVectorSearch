# CandidateVectorSearch

Searching for candidates using sparse matrix + [sparse] vector/matrix multiplication.

Implements the following methods across two DLLs:
- VectorSearch.dll:
  - findTopCandidates: sparse matrix - sparse vector multiplication using [Eigen](https://eigen.tuxfamily.org/).
  - findTopCandidatesBatched: sparse matrix - sparse matrix multiplication using [Eigen](https://eigen.tuxfamily.org/).
- VectorSearchCUDA.dll:
  - findTopCandidates: sparse matrix - dense vector multiplication using [CUDA](https://developer.nvidia.com/cuda-toolkit) ([SpMV](https://docs.nvidia.com/cuda/cusparse/index.html#cusparsespmv)).
  - findTopCandidatesBatched: sparse matrix - sparse matrix multiplication using [CUDA](https://developer.nvidia.com/cuda-toolkit) ([SpGEMM](https://docs.nvidia.com/cuda/cusparse/index.html#cusparsespgemm)).
  - findTopCandidatesBatched2: sparse matrix - dense matrix multiplication using [CUDA](https://developer.nvidia.com/cuda-toolkit) ([SpMM](https://docs.nvidia.com/cuda/cusparse/index.html#cusparsespmm)).
 
VectorSearch.dll implements functions that run on the CPU, while VectorSearchCUDA.dll implements functions that run on a [NVIDIA GPU](https://www.nvidia.com/) using [CUDA](https://developer.nvidia.com/cuda-toolkit) (version [12.2.0_536.25_windows](https://developer.nvidia.com/cuda-toolkit-archive)).

Which functions should be used depends on the problem size and the available hardware. A recommendation is depicted at the figure at the bottom.

## Benchmarks

tbd

## Acknowledgements

- This project uses [Eigen](https://eigen.tuxfamily.org/) and [CUDA](https://developer.nvidia.com/cuda-toolkit) to implement sparse linear algebra, [Eigen](https://eigen.tuxfamily.org/) is licensed under [MPL2](https://www.mozilla.org/en-US/MPL/2.0/), and [CUDA](https://developer.nvidia.com/cuda-toolkit) is owned by [NVIDIA Corporation](https://www.nvidia.com/).
- Special thanks goes to the [Eigen Community Discord](https://discord.gg/2SkEJGqZjR) who helped fixing a bug in the original implementation of `VectorSearch::findTopCandidates`.

## Functions

![Method_Decision_Tree](sparse_alg_decision_tree.png)
