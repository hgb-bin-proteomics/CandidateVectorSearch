# Benchmarks

The following are basically benchmarks of the different sparse matrix/vector
multiplication methods of [Eigen](https://eigen.tuxfamily.org/) and
[cuSPARSE](https://docs.nvidia.com/cuda/cusparse/).

These benchmarks are supposed to be worst-case scenarios when doing candidate
search, e.g. these benchmarks assume that every peptide would yield 100 ions and
every spectrum 1000 peaks, while also performing normalization and gaussian peak
modeling.

We ran benchmarks for different database sizes (different number of candidate
peptides to be considered) to assess how that influences performance of the
different methods. Furthermore, every benchmark is run five times to get a more
comprehensive overview of computation times. The averages are plotted below,
with error bars denoting standard deviation.

For all benchmarks we search 1001 spectra (this is specifically selected to see
if batched multiplication has influence on performance) and return the top 100
candidates. All benchmarks were conducted during light background usage (e.g.
open browser, text editor, etc.).

## System 1 - Standard Office PC

The first system we tested this on was a standard office laptop with the
following hardware:
- Model: Dell Precision 3560
- CPU: Intel Core i7-1185G7 [4 cores @ 1.8 GHz base / 3.0 GHz boost]
- RAM: 16 GB DDR4 RAM [3200 MT/s, NA CAS]
- GPU: Nvidia T500 [2 GB VRAM]
- SSD/HDD: 512 GB NVMe SSD
- OS: Windows 10 Education 64-bit (10.0, Build 19045)

### 10 000 Candidates

`A * B = C where A[10000, 500000] and B[500000, 1001]`

Using a database of 10 000 peptide candidates the methods yield the following
runtimes:

![benchmark_pc_10000](Benchmarks/benchmark_pc_10000.svg)
**Figure 1:** Float32-based sparse matrix * dense vector search using
[Eigen](https://eigen.tuxfamily.org/) yields the fastest computation time of
only 1.02 seconds.

<details><summary>Expand for raw data!</summary>

| Method    |   Candidates |   Run 1 |     Run 2 |    Run 3 |    Run 4 |    Run 5 |      Min |      Max |     Mean |        SD |   Rank |    Y |   N |
|:----------|-------------:|--------:|----------:|---------:|---------:|---------:|---------:|---------:|---------:|----------:|-------:|-----:|----:|
| f32CPU_SV |        10000 | 3.96232 |  3.99317  |  4.16333 |  4.12433 |  4.03925 | 3.96232  |  4.16333 |  4.05648 | 0.0854273 |      8 | 1001 | 100 |
| i32CPU_SV |        10000 | 4.20677 |  4.21627  |  4.18454 |  4.21334 |  4.30658 | 4.18454  |  4.30658 |  4.2255  | 0.0469989 |      9 | 1001 | 100 |
| f32CPU_DV |        10000 | 1.02714 |  0.999038 |  1.04962 |  1.01544 |  1.03139 | 0.999038 |  1.04962 |  1.02453 | 0.0188148 |      1 | 1001 | 100 |
| i32CPU_DV |        10000 | 1.088   |  1.17937  |  1.17109 |  1.1531  |  1.18244 | 1.088    |  1.18244 |  1.1548  | 0.0390465 |      4 | 1001 | 100 |
| f32CPU_SM |        10000 | 1.16123 |  1.14092  |  1.08636 |  1.17035 |  1.1552  | 1.08636  |  1.17035 |  1.14281 | 0.0333204 |      3 | 1001 | 100 |
| i32CPU_SM |        10000 | 1.01817 |  1.06418  |  1.01925 |  1.07144 |  1.13448 | 1.01817  |  1.13448 |  1.0615  | 0.0476856 |      2 | 1001 | 100 |
| f32CPU_DM |        10000 | 1.8242  |  1.77216  |  1.74569 |  1.715   |  1.77249 | 1.715    |  1.8242  |  1.76591 | 0.040254  |      5 | 1001 | 100 |
| i32CPU_DM |        10000 | 1.91169 |  1.86213  |  1.79263 |  1.82148 |  1.81984 | 1.79263  |  1.91169 |  1.84156 | 0.0463954 |      6 | 1001 | 100 |
| f32GPU_DV |        10000 | 4.03647 |  4.09389  |  4.05512 |  4.07632 |  4.05695 | 4.03647  |  4.09389 |  4.06375 | 0.0219723 |      9 | 1001 | 100 |
| f32GPU_DM |        10000 | 3.62518 |  3.74288  |  3.75778 |  3.71924 |  3.73217 | 3.62518  |  3.75778 |  3.71545 | 0.0524091 |      7 | 1001 | 100 |
| f32GPU_SM |        10000 | 9.95502 | 10.0398   | 10.1103  | 10.1644  | 10.0673  | 9.95502  | 10.1644  | 10.0674  | 0.0784879 |     11 | 1001 | 100 |

</details>

### 100 000 Candidates

`A * B = C where A[100000, 500000] and B[500000, 1001]`

Using a database of 100 000 peptide candidates the methods yield the following
runtimes:

![benchmark_pc_100000](Benchmarks/benchmark_pc_100000.svg)
**Figure 2:** Float32-based sparse matrix * sparse matrix search using
[Eigen](https://eigen.tuxfamily.org/) yields the fastest computation time of
only 5.08 seconds. Note that `f32GPU_SM` has been excluded from the plot since
computation times exceeded all other methods by more than 10-fold. The raw data
is available below.

<details><summary>Expand for raw data!</summary>

| Method    |   Candidates |     Run 1 |     Run 2 |     Run 3 |     Run 4 |     Run 5 |       Min |       Max |      Mean |        SD |   Rank |    Y |   N |
|:----------|-------------:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|-------:|-----:|----:|
| f32CPU_SV |       100000 |  35.304   |  34.8771  |  31.7219  |  33.6381  |  28.2473  |  28.2473  |  35.304   |  32.7577  |  2.87956  |      9 | 1001 | 100 |
| i32CPU_SV |       100000 |  41.3168  |  42.1746  |  35.1852  |  33.7421  |  31.1516  |  31.1516  |  42.1746  |  36.7141  |  4.82477  |     10 | 1001 | 100 |
| f32CPU_DV |       100000 |   9.8869  |   9.8668  |   9.57659 |   7.34046 |   6.65369 |   6.65369 |   9.8869  |   8.66489 |  1.54662  |      3 | 1001 | 100 |
| i32CPU_DV |       100000 |   9.78072 |   9.80233 |   9.30471 |   7.98685 |   9.94904 |   7.98685 |   9.94904 |   9.36473 |  0.807484 |      4 | 1001 | 100 |
| f32CPU_SM |       100000 |   5.92302 |   5.56398 |   4.88576 |   4.40602 |   4.63187 |   4.40602 |   5.92302 |   5.08213 |  0.639863 |      1 | 1001 | 100 |
| i32CPU_SM |       100000 |   5.36173 |   5.56918 |   5.83226 |   4.43903 |   4.73719 |   4.43903 |   5.83226 |   5.18788 |  0.581964 |      2 | 1001 | 100 |
| f32CPU_DM |       100000 |  13.9166  |  14.7445  |  14.933   |  11.0524  |  11.2453  |  11.0524  |  14.933   |  13.1783  |  1.89294  |      6 | 1001 | 100 |
| i32CPU_DM |       100000 |  14.0893  |  14.2498  |  14.913   |  11.2577  |  10.6276  |  10.6276  |  14.913   |  13.0275  |  1.9409   |      5 | 1001 | 100 |
| f32GPU_DV |       100000 |  19.6112  |  20.1476  |  19.9083  |  19.1877  |  18.8013  |  18.8013  |  20.1476  |  19.5312  |  0.542965 |      7 | 1001 | 100 |
| f32GPU_DM |       100000 |  26.7439  |  26.9163  |  26.7714  |  26.4168  |  26.3571  |  26.3571  |  26.9163  |  26.6411  |  0.241999 |      8 | 1001 | 100 |
| f32GPU_SM |       100000 | 880.093   | 919.047   | 807.312   | 792.249   | 774.371   | 774.371   | 919.047   | 834.615   | 61.9812   |     11 | 1001 | 100 |

</details>

### 1 000 000 Candidates

`A * B = C where A[1000000, 500000] and B[500000, 1001]`

Using a database of 1 000 000 peptide candidates the methods yield the following
runtimes:

![benchmark_pc_1000000](Benchmarks/benchmark_pc_1000000.svg)
**Figure 3:** Int32-based sparse matrix * sparse matrix search using
[Eigen](https://eigen.tuxfamily.org/) yields the fastest computation time of
only 45.04 seconds. Note that `f32GPU_SM` has been excluded from the plot since
the method ran out of memory. The raw data is available below.

<details><summary>Expand for raw data!</summary>

| Method    |   Candidates |    Run 1 |    Run 2 |    Run 3 |    Run 4 |    Run 5 |      Min |      Max |     Mean |       SD |   Rank |    Y |   N |
|:----------|-------------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|-------:|-----:|----:|
| f32CPU_SV |      1000000 | 292.024  | 305.725  | 275.855  | 283.298  | 308.572  | 275.855  | 308.572  | 293.095  | 14.0837  |      9 | 1001 | 100 |
| i32CPU_SV |      1000000 | 337.896  | 330.922  | 293.953  | 328.767  | 351.387  | 293.953  | 351.387  | 328.585  | 21.2805  |     10 | 1001 | 100 |
| f32CPU_DV |      1000000 |  87.1387 |  78.9427 |  73.9562 |  82.5062 |  88.0868 |  73.9562 |  88.0868 |  82.1261 |  5.8669  |      4 | 1001 | 100 |
| i32CPU_DV |      1000000 |  88.2644 |  76.9449 |  70.4682 |  81.2829 |  92.4659 |  70.4682 |  92.4659 |  81.8853 |  8.77158 |      3 | 1001 | 100 |
| f32CPU_SM |      1000000 |  59.1796 |  42.2678 |  38.3327 |  43.238  |  61.2774 |  38.3327 |  61.2774 |  48.8591 | 10.5662  |      2 | 1001 | 100 |
| i32CPU_SM |      1000000 |  41.7158 |  45.5913 |  38.0705 |  43.2118 |  56.596  |  38.0705 |  56.596  |  45.0371 |  7.0145  |      1 | 1001 | 100 |
| f32CPU_DM |      1000000 | 105.11   | 106.617  |  95.4387 | 105.418  | 114.833  |  95.4387 | 114.833  | 105.483  |  6.88718 |      5 | 1001 | 100 |
| i32CPU_DM |      1000000 | 113.402  | 105.918  |  96.2186 | 109.205  | 113.995  |  96.2186 | 113.995  | 107.748  |  7.23534 |      6 | 1001 | 100 |
| f32GPU_DV |      1000000 | 166.4    | 165.727  | 165.672  | 167.374  | 169.121  | 165.672  | 169.121  | 166.859  |  1.4387  |      7 | 1001 | 100 |
| f32GPU_DM |      1000000 | 301.266  | 299.136  | 254.768  | 257.244  | 256.796  | 254.768  | 301.266  | 273.842  | 24.0922  |      8 | 1001 | 100 |

</details>

### 5 000 000 Candidates

`A * B = C where A[5000000, 500000] and B[500000, 1001]`

Using a database of 5 000 000 peptide candidates the methods yield the following
runtimes:

![benchmark_pc_5000000](Benchmarks/benchmark_pc_5000000.svg)
**Figure 5:** Float32-based sparse matrix * sparse matrix search using
[Eigen](https://eigen.tuxfamily.org/) yields the fastest computation time of
only 210.98 seconds. Note that all GPU-based methods have been excluded from the
plot since their computation times exceeded that of CPU-based methods by more
than 10-fold or because they ran out of memory. The raw data is available
below.

<details><summary>Expand for raw data!</summary>

| Method    |   Candidates |     Run 1 |     Run 2 |     Run 3 |     Run 4 |     Run 5 |       Min |       Max |      Mean |       SD |   Rank |    Y |   N |
|:----------|-------------:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|---------:|-------:|-----:|----:|
| f32CPU_SV |      5000000 |  1488.95  |  1753.58  |  1409.96  |  1405.52  |  1433.23  |  1405.52  |  1753.58  |  1498.25  | 146.545  |      7 | 1001 | 100 |
| i32CPU_SV |      5000000 |  1456.77  |  2199.68  |  1443.93  |  1433.08  |  1640.18  |  1433.08  |  2199.68  |  1634.73  | 327.082  |      8 | 1001 | 100 |
| f32CPU_DV |      5000000 |   362.758 |   434.276 |   371.356 |   371.242 |   396.402 |   362.758 |   434.276 |   387.207 |  29.1716 |      4 | 1001 | 100 |
| i32CPU_DV |      5000000 |   360.054 |   429.113 |   362.396 |   354.354 |   383.947 |   354.354 |   429.113 |   377.973 |  30.7108 |      3 | 1001 | 100 |
| f32CPU_SM |      5000000 |   202.057 |   253.796 |   195.927 |   197.155 |   205.942 |   195.927 |   253.796 |   210.975 |  24.2692 |      1 | 1001 | 100 |
| i32CPU_SM |      5000000 |   196.972 |   247.733 |   238.983 |   217.433 |   192.904 |   192.904 |   247.733 |   218.805 |  24.4611 |      2 | 1001 | 100 |
| f32CPU_DM |      5000000 |   495.787 |   543.992 |   501.467 |   506.691 |   542.057 |   495.787 |   543.992 |   517.999 |  23.1783 |      5 | 1001 | 100 |
| i32CPU_DM |      5000000 |   494.032 |   519.314 |   542.015 |   496.312 |   542.956 |   494.032 |   542.956 |   518.926 |  23.6736 |      6 | 1001 | 100 |
| f32GPU_DV |      5000000 | 13753.4   | 13738.6   | 13777.2   | 13396.6   | 14214     | 13396.6   | 14214     | 13775.9   | 290.558  |      9 | 1001 | 100 |
| f32GPU_DM |      5000000 | 14965.1   | 15271.3   | 15013.6   | 14908.9   | 14943.8   | 14908.9   | 15271.3   | 15020.5   | 145.243  |     10 | 1001 | 100 |

</details>

## System 2 - High Performance PC

The second system we tested this on was a more powerful desktop PC with the
following (more recent) hardware:
- MB: ASUS ROG Strix B650E-I
- CPU: AMD Ryzen 7900X [12 cores @ 4.7 GHz base / 5.6 GHz boost]
- RAM: Kingston 64 GB DDR5 RAM [5600 MT/s, 36 CAS]
- GPU: ASUS Dual [Nvidia] GeForce RTX 4060 Ti OC [16 GB VRAM]*
- SSD/HDD: Corsair MP600 Pro NH 2 TB NVMe SSD [PCIe 4.0]
- OS: Windows 11 Pro 64-bit (, Build)

*_Note:_ `Dual` _is part of the name, this is a single graphics card!_

### 10 000 Candidates

`A * B = C where A[10000, 500000] and B[500000, 1001]`

Using a database of 10 000 peptide candidates the methods yield the following
runtimes:
