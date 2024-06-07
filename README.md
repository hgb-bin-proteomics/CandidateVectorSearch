![test_state_windows](https://github.com/hgb-bin-proteomics/CandidateVectorSearch/workflows/Test%20for%20Windows/badge.svg)
![test_state_ubuntu](https://github.com/hgb-bin-proteomics/CandidateVectorSearch/workflows/Test%20for%20Ubuntu%2022.04/badge.svg)
![test_state_linux](https://github.com/hgb-bin-proteomics/CandidateVectorSearch/workflows/Test%20for%20generic%20Linux/badge.svg)
![test_state_macos](https://github.com/hgb-bin-proteomics/CandidateVectorSearch/workflows/Test%20for%20macOS/badge.svg)

# CandidateVectorSearch

Searching for peptide candidates using sparse matrix + [sparse] vector/matrix multiplication. This is the computational backend for
[CandidateSearch](https://github.com/hgb-bin-proteomics/CandidateSearch) - a search engine that aims to (quickly) identify peptide candidates for
a given mass spectrum without any information about precursor mass or variable modifications. This is also the computational backend for the
non-cleavable crosslink search in [MS Annika](https://github.com/hgb-bin-proteomics/MSAnnika).

Implements the following methods across two DLLs:
- [VectorSearch.dll](https://github.com/hgb-bin-proteomics/CandidateVectorSearch/blob/master/VectorSearch/dllmain.cpp):
  - findTopCandidates: sparse matrix - sparse vector multiplication [f32] using [Eigen](https://eigen.tuxfamily.org/).
  - findTopCandidatesInt: sparse matrix - sparse vector multiplication [i32] using [Eigen](https://eigen.tuxfamily.org/).
  - findTopCandidates2: sparse matrix - dense vector multiplication [f32] using [Eigen](https://eigen.tuxfamily.org/).
  - findTopCandidates2Int: sparse matrix - dense vector multiplication [i32] using [Eigen](https://eigen.tuxfamily.org/).
  - findTopCandidatesBatched: sparse matrix - sparse matrix multiplication [f32] using [Eigen](https://eigen.tuxfamily.org/).
  - findTopCandidatesBatchedInt: sparse matrix - sparse matrix multiplication [i32] using [Eigen](https://eigen.tuxfamily.org/).
  - findTopCandidatesBatched2: sparse matrix - dense matrix multiplication [f32] using [Eigen](https://eigen.tuxfamily.org/).
  - findTopCandidatesBatched2Int: sparse matrix - dense matrix multiplication [i32] using [Eigen](https://eigen.tuxfamily.org/).
- [VectorSearchCUDA.dll](https://github.com/hgb-bin-proteomics/CandidateVectorSearch/blob/master/VectorSearchCUDA/dllmain.cpp):
  - findTopCandidatesCuda: sparse matrix - dense vector multiplication [f32] using [CUDA](https://developer.nvidia.com/cuda-toolkit) ([SpMV](https://docs.nvidia.com/cuda/cusparse/index.html#cusparsespmv)).
  - findTopCandidatesCudaBatched: sparse matrix - sparse matrix multiplication [f32] using [CUDA](https://developer.nvidia.com/cuda-toolkit) ([SpGEMM](https://docs.nvidia.com/cuda/cusparse/index.html#cusparsespgemm)).
  - findTopCandidatesCudaBatched2: sparse matrix - dense matrix multiplication [f32] using [CUDA](https://developer.nvidia.com/cuda-toolkit) ([SpMM](https://docs.nvidia.com/cuda/cusparse/index.html#cusparsespmm)).

VectorSearch.dll implements functions that run on the CPU, while VectorSearchCUDA.dll implements functions that run on a [NVIDIA GPU](https://www.nvidia.com/) using [CUDA](https://developer.nvidia.com/cuda-toolkit) (version [12.2.0_536.25_windows](https://developer.nvidia.com/cuda-toolkit-archive)).

Which functions should be used depends on the problem size and the available hardware. A general recommendation is to use
`findTopCanidates2` or `findTopCandidates2Int` on CPUs and `findTopCandidatesCuda` on GPUs.

## Documentation

Functions are documented within the source code:
- [VectorSearch.dll](https://github.com/hgb-bin-proteomics/CandidateVectorSearch/blob/master/VectorSearch/dllmain.cpp)
- [VectorSearchCUDA.dll](https://github.com/hgb-bin-proteomics/CandidateVectorSearch/blob/master/VectorSearchCUDA/dllmain.cpp)

A better description of the input arrays is given in [input.md](input.md).

An example usage where functions are called from a C# application is given in
[here (CPU)](https://github.com/hgb-bin-proteomics/CandidateSearch/blob/master/CandidateSearchCPU.cs)
and [here (GPU)](https://github.com/hgb-bin-proteomics/CandidateSearch/blob/master/CandidateSearchGPU.cs).
A wrapper for C# is given in [here](https://github.com/hgb-bin-proteomics/CandidateVectorSearch/blob/master/VectorSearchInterface/VectorSearchAPI.cs).

Documentation is also available on
[https://hgb-bin-proteomics.github.io/CandidateVectorSearch/](https://hgb-bin-proteomics.github.io/CandidateVectorSearch/).

## Benchmarks

See [benchmarks.md](benchmarks.md).

## Requirements

- [.NET](https://dotnet.microsoft.com/en-us/) may be required on some systems to run the `DataLoader` testing suite.
- \[Optional\] Using GPU based approaches (e.g. anything implemented in `VectorSearchCUDA.dll`) requires a CUDA capable GPU and CUDA version == 12.2.0
([download here](https://developer.nvidia.com/cuda-12-2-0-download-archive)). Other CUDA versions may or may not produce the desired results
([see this issue](https://github.com/hgb-bin-proteomics/CandidateVectorSearch/issues/32)).

## Downloads

Compiled DLLs are available in the `dll` folder or in
[Releases](https://github.com/hgb-bin-proteomics/CandidateVectorSearch/releases).

We supply compiled executables and DLLs for:
- Windows 10/11 (x86, 64-bit)
- Ubuntu 22.04 (x86, 64-bit)
- macOS 14.4 (arm, 64-bit)

For other operating systems/architectures please compile the source code yourself! Exemplary compilation instructions for linux and macOS can
be found here: [linux.md](linux.md) and [macos.md](macos.md).

## Limitations

Please be aware of the following limitations:
- Ions/peaks up to 5000 m/z are supported, beyond that they are discarded.
- The encoding precision is 0.01 (m/z, Dalton).
- Only matrices up to 2 * 10<sup>9</sup> non-zero elements are supported \[see [this issue](https://github.com/hgb-bin-proteomics/CandidateVectorSearch/issues/42)\].
- \[Eigen\]\[Sparse\] Sparse candidate matrices support up to 100 elements per row, beyond that matrix creation might be slow due to resizing.
  - This means every peptide candidate can be encoded up to 100 ions.
- \[Eigen\]\[Sparse\] Sparse spectrum matrices support up to 1000 elements per row, beyond that matrix creation might be slow due to resizing.
  - This means spectra with more than 1000 peaks should be deisotoped, deconvoluted or peak picked to decrease the number of peaks.
  - This does not affect dense spectrum matrices.
- \[Eigen\]\[i32\] The rounding precision of converting floats to integers is 0.001, the exact rounding for a float `val` is `(int) round(val * 1000.0f)`.
- \[Eigen\]\[i32\] Integer based methods do not allow tolerances below 0.01 because they might cause overflows.
- \[CUDA\] Sparse matrix - sparse matrix multiplication tends to be very slow and very memory hungry, most likely caused by memory overhead and the output matrix not being sparse.

## Implementing your own matrix products

If you want to implement your own (and hopefully faster) computation for matrix products, we offer a template repository that walks you through that:
[CandidateVectorSearch_template](https://github.com/hgb-bin-proteomics/CandidateVectorSearch_template/)

## Acknowledgements

- This project uses [Eigen](https://eigen.tuxfamily.org/) and [CUDA](https://developer.nvidia.com/cuda-toolkit) to implement sparse linear algebra, [Eigen](https://eigen.tuxfamily.org/) is licensed under [MPL2](https://www.mozilla.org/en-US/MPL/2.0/), and [CUDA](https://developer.nvidia.com/cuda-toolkit) is owned by [NVIDIA Corporation](https://www.nvidia.com/).
- Special thanks goes to the [Eigen Community Discord](https://discord.gg/2SkEJGqZjR) who helped fixing a bug in the original implementation of `VectorSearch::findTopCandidates`.

## Citing

If you are using [parts of] *CandidateVectorSearch* please cite:

```
MS Annika 3.0 (publication wip)
```

## License

- [MIT](https://github.com/hgb-bin-proteomics/CandidateVectorSearch/blob/master/LICENSE)

## Contact

[micha.birklbauer@fh-hagenberg.at](mailto:micha.birklbauer@fh-hagenberg.at)
