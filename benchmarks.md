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
comprehensive overview of computation times.

For all benchmarks we search 1001 spectra (this is specifically selected to see
if batched multiplication has influence on performance) and return the top 100
candidates.

## System 1 - Standard Office PC

The first system we tested this on was a standard office laptop with the
following hardware:
- Model: Dell Precision 3560
- CPU: Intel Core i7-1185G7 [4 cores @ 1.8 GHz base/ 3.0 GHz boost]
- RAM: 16 GB DDR4 RAM [3200 MT/s, NA CAS]
- GPU: Nvidia T500 [2GB VRAM]
- SSD/HDD: 512 GB NVMe SSD
- OS: Windows 10 Education 64-bit (10.0, Build 19045)

### 10 000 Candidates

`A * B = C where A[10000, 500000] and B[500000, 1001]`

Using a database of 10 000 peptide candidates the methods yield the following
runtimes:

![benchmark_pc_10000](Benchmarks/benchmark_pc_10000.svg)
**Figure 1:** Caption.


<details><summar>Expand for raw data!</summary>

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
