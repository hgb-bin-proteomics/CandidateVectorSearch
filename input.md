# CandidateVectorSearch Input Arrays

All `findTopCandidates` functions require four integer arrays as input:
- For the CPU versions they are called:
  - `candidatesValues`
  - `candidatesIdx`
  - `spectraValues`
  - `spectraIdx`
- For the GPU versions they are called:
  - `csrRowoffsets`
  - `csrColIdx`
  - `spectraValues`
  - `spectraIdx`
Note that `spectraValues` and `spectraIdx` is the same for both versions. The
following should highlight how to get these arrays.

Consider two peptides with the following ion m/z values:

```python
peptide_1 = [321.3251, 531.7851, 556.2134, 643.9867, 989.9911]
peptide_2 = [301.4156, 411.6598, 713.7981, 754.3412, 811.9812, 871.4351]
```

And the following two spectra with m/z values as given:

```python
spectrum_1 = [135.7413, 321.3251, 531.7851, 989.9911]
spectrum_2 = [101.8931, 301.4156, 713.7981, 754.3412, 811.9812, 871.4351]
```

## candidatesValues and csrColIdx

The `candidatesValues` and `csrColIdx` array are the same. They contain all ion
m/z values from all peptides concatenated. The values should be multiplied by
100 and rounded to the nearest integer. Note that *CandidateVectorSearch* only
supports m/z values up to 5000 so anything beyond that should be discarded!

```python
candidatesValues = [32133, 53179, 55621, 64399, 98999,
                    30142, 41166, 71380, 75434, 81198, 87144]
csrColIdx = candidatesValues
```

## candidatesIdx

The `candidatesIdx` array indicates the starting position of every peptide in
`candidatesValues`. It always contains as many elements as there are peptides!

```python
candidatesIdx = [0, 5]
```

## csrRowoffsets

Analogous to `candidatesIdx` the array `csrRowoffsets` indicates start and end
positions of all peptides in the `csrColIdx` array. This follows the CSR format
as described [here](https://docs.nvidia.com/cuda/cusparse/index.html#compressed-sparse-row-csr).
The `csrRowoffsets` array always contains number of peptides + 1 elements!

```python
csrRowoffsets = [0, 5, 11]
```

## spectraValues

The `spectraValues` array contains all m/z values of all peaks of all spectra.
Again the values should be multiplied by 100 and rounded to the nearest integer.
Peaks with m/z values greater than 5000 should be discarded.

```python
spectraValues = [13574, 32133, 53179, 98999]
                 10189, 30142, 71380, 75434, 81198, 87144]
```

## spectraIdx

The `spectraIdx` array is analogous to the `candidatesIdx` array and contains
the indices of where every spectrum starts in `spectraValues`. It always
contains as many elements as there are spectra.

```python
spectraIdx = [0, 4]
```

## Code example

A code example where functions are called from a C# application is given in
[here (CPU)](https://github.com/hgb-bin-proteomics/CandidateSearch/blob/master/CandidateSearchCPU.cs)
and [here (GPU)](https://github.com/hgb-bin-proteomics/CandidateSearch/blob/master/CandidateSearchGPU.cs).
