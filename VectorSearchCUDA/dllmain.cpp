// dllmain.cpp : Defines the entry point for the DLL application.
#include "pch.h"
#include <cuda_runtime_api.h>
#include <cusparse.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <vector>

const int versionMajor = 1;
const int versionMinor = 3;
const int versionFix = 5;

#define METHOD_EXPORTS
#ifdef METHOD_EXPORTS
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __declspec(dllimport)
#endif

const int MASS_RANGE = 1300;
const int MASS_MULTIPLIER = 100;
const int ENCODING_SIZE = MASS_RANGE * MASS_MULTIPLIER;
const double ONE_OVER_SQRT_PI = 0.39894228040143267793994605993438;

extern "C" {
    EXPORT int* findTopCandidatesCuda(int*, int*, 
                                      int*, int*,
                                      int, int, 
                                      int, int,
                                      int, float,
                                      bool, bool,
                                      int);

    EXPORT int* findTopCandidatesCudaBatched(int*, int*,
                                             int*, int*,
                                             int, int,
                                             int, int,
                                             int, float,
                                             bool, bool,
                                             int,
                                             int);

    EXPORT int releaseMemoryCuda(int*);
}

float squared(float);
float normpdf(float, float, float);
int getRowIdx(int*, int, int);

int CHECK_CUDA(cudaError_t);
int CHECK_CUSPARSE(cusparseStatus_t);

int CHECK_CUDA(cudaError_t status)
{
    if (status != cudaSuccess) {
        printf("CUDA API failed at line %d with error: %s (%d)\n",
            __LINE__, cudaGetErrorString(status), status);
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

int CHECK_CUSPARSE(cusparseStatus_t status)
{
    if (status != CUSPARSE_STATUS_SUCCESS) {
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",
            __LINE__, cusparseGetErrorString(status), status);
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

/// <summary>
/// A function that calculates the top n candidates for each spectrum.
/// </summary>
/// <param name="csrRowoffsets">Rowoffsets (int array) of the CSR sparse matrix (L: rows + 1).</param>
/// <param name="csrColIdx">Column indices (int array) of the CSR sparse matrix (L: NNZ).</param>
/// <param name="spectraValues">An integer array of peaks from experimental spectra flattened.</param>
/// <param name="spectraIdx">An integer array that contains indices of where each spectrum starts in spectraValues.</param>
/// <param name="csrRowoffsetsLength">Length (int) of csrRowoffsets (rows + 1).</param>
/// <param name="csrNNZ">Number of non-zero entries (int) in the CSR sparse matrix.</param>
/// <param name="sVLength">Length (int) of spectraValues.</param>
/// <param name="sILength">Length (int) of spectraIdx.</param>
/// <param name="n">How many of the best hits should be returned (int).</param>
/// <param name="tolerance">Tolerance for peak matching (float).</param>
/// <param name="normalize">If candidate vectors should be normalized to sum(elements) = 1 (bool).</param>
/// <param name="gaussianTol">If spectrum peaks should be modelled as normal distributions or not (bool).</param>
/// <param name="verbose">Print info every (int) processed spectra.</param>
/// <returns>An integer array of length sILength * n containing the indexes of the top n candidates for each spectrum.</returns>
int* findTopCandidatesCuda(int* csrRowoffsets, int* csrColIdx, 
                           int* spectraValues, int* spectraIdx,
                           int csrRowoffsetsLength, int csrNNZ,
                           int sVLength, int sILength,
                           int n, float tolerance,
                           bool normalize, bool gaussianTol,
                           int verbose) {

    if (n >= csrRowoffsetsLength) {
        throw std::invalid_argument("Cannot return more hits than number of candidates!");
    }

    std::cout << "Running CUDA vector search version " << versionMajor << "." << versionMinor << "." << versionFix << std::endl;

    float t = round(tolerance * MASS_MULTIPLIER);
    int* result = new int[sILength * n];
    float* csrValues = new float[csrNNZ];
    float* MVresult = new float[csrRowoffsetsLength - 1] {0.0};

    // create csrValues
    for (int i = 0; i < csrRowoffsetsLength - 1; ++i) {
        int startIter = csrRowoffsets[i];
        int endIter = csrRowoffsets[i + 1];
        int nrNonZero = endIter - startIter;
        float val = normalize ? 1.0 / (float) nrNonZero : 1.0;
        for (int j = startIter; j < endIter; ++j) {
            csrValues[j] = val;
        }
    }

    // cusparse spmv variables
    float alpha = 1.0;
    float beta = 0.0;

    // device memory management
    int* dM_csrRowoffsets;
    int* dM_csrColIdx;
    float* dM_csrValues;
    float* dV;
    float* dMVresult;

    CHECK_CUDA(cudaMalloc((void**) &dM_csrRowoffsets, csrRowoffsetsLength * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**) &dM_csrColIdx, csrNNZ * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**) &dM_csrValues, csrNNZ * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**) &dV, ENCODING_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**) &dMVresult, (csrRowoffsetsLength - 1) * sizeof(float)));

    // device memory copy
    CHECK_CUDA(cudaMemcpy(dM_csrRowoffsets, csrRowoffsets, csrRowoffsetsLength * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dM_csrColIdx, csrColIdx, csrNNZ * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dM_csrValues, csrValues, csrNNZ * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dMVresult, MVresult, (csrRowoffsetsLength - 1) * sizeof(float), cudaMemcpyHostToDevice));

    // device setup cusparse
    cusparseHandle_t handle = NULL;
    cusparseSpMatDescr_t mat;
    cusparseDnVecDescr_t vec;
    cusparseDnVecDescr_t res;
    void* dBuffer = NULL;
    size_t bufferSize = 0;

    CHECK_CUSPARSE(cusparseCreate(&handle));
    CHECK_CUSPARSE(cusparseCreateCsr(&mat, csrRowoffsetsLength - 1, ENCODING_SIZE, csrNNZ,
                                     dM_csrRowoffsets, dM_csrColIdx, dM_csrValues,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&res, csrRowoffsetsLength - 1, dMVresult, CUDA_R_32F));

    // device iterative spmv
    for (int i = 0; i < sILength; ++i) {

        float* V = new float[ENCODING_SIZE] {0.0};

        // host spectrum encoding
        int startIter = spectraIdx[i];
        int endIter = i + 1 == sILength ? sVLength : spectraIdx[i + 1];
        for (int j = startIter; j < endIter; ++j) {
            auto currentPeak = spectraValues[j];
            auto minPeak = currentPeak - t > 0 ? currentPeak - t : 0;
            auto maxPeak = currentPeak + t < ENCODING_SIZE ? currentPeak + t : ENCODING_SIZE - 1;

            for (int k = minPeak; k <= maxPeak; ++k) {
                float currentVal = V[k];
                float newVal = gaussianTol ? normpdf((float) k, (float) currentPeak, (float) (t / 3.0)) : 1.0;
                V[k] = max(currentVal, newVal);
            }
        }

        // device spmv
        CHECK_CUDA(cudaMemcpy(dV, V, ENCODING_SIZE * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUSPARSE(cusparseCreateDnVec(&vec, ENCODING_SIZE, dV, CUDA_R_32F));
        CHECK_CUSPARSE(cusparseSpMV_bufferSize(handle,
                                               CUSPARSE_OPERATION_NON_TRANSPOSE,
                                               &alpha, mat, vec, &beta, res,
                                               CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
        CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));
        CHECK_CUSPARSE(cusparseSpMV(handle,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha, mat, vec, &beta, res,
                                    CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));

        CHECK_CUDA(cudaMemcpy(MVresult, dMVresult, (csrRowoffsetsLength - 1) * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUSPARSE(cusparseDestroyDnVec(vec));
        delete[] V;

        // host result max
        auto* idx = new int[csrRowoffsetsLength - 1];
        std::iota(idx, idx + csrRowoffsetsLength - 1, 0);
        std::sort(idx, idx + csrRowoffsetsLength - 1, [&](int i, int j) {return MVresult[i] > MVresult[j];});

        for (int j = 0; j < n; ++j) {
            result[i * n + j] = idx[j];
        }

        delete[] idx;

        if (verbose != 0 && (i + 1) % verbose == 0) {
            std::cout << "Searched " << i + 1 << " spectra in total..." << std::endl;
        }
    }

    // device destroy descriptors
    CHECK_CUSPARSE(cusparseDestroySpMat(mat));
    CHECK_CUSPARSE(cusparseDestroyDnVec(res));
    CHECK_CUSPARSE(cusparseDestroy(handle));

    // device memory deallocation
    CHECK_CUDA(cudaFree(dBuffer));
    CHECK_CUDA(cudaFree(dM_csrRowoffsets));
    CHECK_CUDA(cudaFree(dM_csrColIdx));
    CHECK_CUDA(cudaFree(dM_csrValues));
    CHECK_CUDA(cudaFree(dV));
    CHECK_CUDA(cudaFree(dMVresult));

    // host memory deallocation
    delete[] MVresult;
    delete[] csrValues;

    return result;
}

/// <summary>
/// A function that calculates the top n candidates for each spectrum. Uses cusparseSpGEMM to calculate matrix product.
/// </summary>
/// <param name="csrRowoffsets">Rowoffsets (int array) of the CSR sparse matrix (L: rows + 1).</param>
/// <param name="csrColIdx">Column indices (int array) of the CSR sparse matrix (L: NNZ).</param>
/// <param name="spectraValues">An integer array of peaks from experimental spectra flattened.</param>
/// <param name="spectraIdx">An integer array that contains indices of where each spectrum starts in spectraValues.</param>
/// <param name="csrRowoffsetsLength">Length (int) of csrRowoffsets (rows + 1).</param>
/// <param name="csrNNZ">Number of non-zero entries (int) in the CSR sparse matrix.</param>
/// <param name="sVLength">Length (int) of spectraValues.</param>
/// <param name="sILength">Length (int) of spectraIdx.</param>
/// <param name="n">How many of the best hits should be returned (int).</param>
/// <param name="tolerance">Tolerance for peak matching (float).</param>
/// <param name="normalize">If candidate vectors should be normalized to sum(elements) = 1 (bool).</param>
/// <param name="gaussianTol">If spectrum peaks should be modelled as normal distributions or not (bool).</param>
/// <param name="batchSize">How many spectra (int) should be searched at once.</param>
/// <param name="verbose">Print info every (int) processed spectra.</param>
/// <returns>An integer array of length sILength * n containing the indexes of the top n candidates for each spectrum.</returns>
int* findTopCandidatesCudaBatched(int* csrRowoffsets, int* csrColIdx,
                                  int* spectraValues, int* spectraIdx,
                                  int csrRowoffsetsLength, int csrNNZ,
                                  int sVLength, int sILength,
                                  int n, float tolerance,
                                  bool normalize, bool gaussianTol,
                                  int batchSize,
                                  int verbose) {

    if (n >= csrRowoffsetsLength) {
        throw std::invalid_argument("Cannot return more hits than number of candidates!");
    }

    std::cout << "Running CUDA sparse matrix search version " << versionMajor << "." << versionMinor << "." << versionFix << std::endl;

    float t = round(tolerance * MASS_MULTIPLIER);
    int* result = new int[sILength * n];
    float* csrValues = new float[csrNNZ];

    // create csrValues
    for (int i = 0; i < csrRowoffsetsLength - 1; ++i) {
        int startIter = csrRowoffsets[i];
        int endIter = csrRowoffsets[i + 1];
        int nrNonZero = endIter - startIter;
        float val = normalize ? 1.0 / (float) nrNonZero : 1.0;
        for (int j = startIter; j < endIter; ++j) {
            csrValues[j] = val;
        }
    }

    // cusparse spgemM variables
    float alpha = 1.0;
    float beta = 0.0;

    // device memory management
    int* dm_csrRowoffsets;
    int* dm_csrColIdx;
    float* dm_csrValues;
    int* dM_csrRowoffsets;
    int* dM_csrColIdx;
    float* dM_csrValues;
    int* dspgemM_csrRowoffsets;
    int* dspgemM_csrColIdx;
    float* dspgemM_csrValues;

    // device buffer managment
    void* dBuffer1 = NULL;
    void* dBuffer2 = NULL;
    size_t bufferSize1 = 0;
    size_t bufferSize2 = 0;

    // device m memory management
    CHECK_CUDA(cudaMalloc((void**) &dm_csrRowoffsets, csrRowoffsetsLength * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**) &dm_csrColIdx, csrNNZ * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**) &dm_csrValues, csrNNZ * sizeof(float)));

    // device m memory copy
    CHECK_CUDA(cudaMemcpy(dm_csrRowoffsets, csrRowoffsets, csrRowoffsetsLength * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dm_csrColIdx, csrColIdx, csrNNZ * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dm_csrValues, csrValues, csrNNZ * sizeof(float), cudaMemcpyHostToDevice));

    // device setup cusparse m
    cusparseHandle_t handle = NULL;
    cusparseSpMatDescr_t mat;

    CHECK_CUSPARSE(cusparseCreate(&handle));
    CHECK_CUSPARSE(cusparseCreateCsr(&mat, csrRowoffsetsLength - 1, ENCODING_SIZE, csrNNZ,
                                     dm_csrRowoffsets, dm_csrColIdx, dm_csrValues,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

    // device allocate csr rowoffsets of spgemM result
    CHECK_CUDA(cudaMalloc((void**) &dspgemM_csrRowoffsets, csrRowoffsetsLength * sizeof(int)));

    for (int i = 0; i < sILength; i += batchSize) {

        // calculate csr representation of spectrum matrix
        std::vector<std::vector<float>> vectorStorage;
        std::vector<int> M_csrRowoffsets;
        std::vector<int> M_csrColIdx;
        std::vector<float> M_csrValues;

        M_csrRowoffsets.push_back(0);

        for (int s = 0; s < batchSize; ++s) {

            std::vector<float> v(ENCODING_SIZE, 0.0);

            if (i + s < sILength) {

                int startIter = spectraIdx[i + s];
                int endIter = i + s + 1 == sILength ? sVLength : spectraIdx[i + s + 1];
                
                for (int j = startIter; j < endIter; ++j) {
                    auto currentPeak = spectraValues[j];
                    auto minPeak = currentPeak - t > 0 ? currentPeak - t : 0;
                    auto maxPeak = currentPeak + t < ENCODING_SIZE ? currentPeak + t : ENCODING_SIZE - 1;

                    for (int k = minPeak; k <= maxPeak; ++k) {
                        float currentVal = v[k];
                        float newVal = gaussianTol ? normpdf((float) k, (float) currentPeak, (float) (t / 3.0)) : 1.0;
                        v[k] = max(currentVal, newVal);
                    }
                }
            }

            vectorStorage.push_back(v);
        }

        int M_csrNNZ = 0;
        for (int j = 0; j < ENCODING_SIZE; ++j) {
            for (int s = 0; s < batchSize; ++s) {
                if (vectorStorage[s][j] != 0.0) {
                    M_csrColIdx.push_back(s);
                    M_csrValues.push_back(vectorStorage[s][j]);
                    ++M_csrNNZ;
                }
            }
            M_csrRowoffsets.push_back(M_csrNNZ);
        }

        vectorStorage.clear();

        // device M memory management
        CHECK_CUDA(cudaMalloc((void**) &dM_csrRowoffsets, (ENCODING_SIZE + 1) * sizeof(int)));
        CHECK_CUDA(cudaMalloc((void**) &dM_csrColIdx, M_csrNNZ * sizeof(int)));
        CHECK_CUDA(cudaMalloc((void**) &dM_csrValues, M_csrNNZ * sizeof(float)));

        // device m memory copy
        CHECK_CUDA(cudaMemcpy(dM_csrRowoffsets, M_csrRowoffsets.data(), (ENCODING_SIZE + 1) * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(dM_csrColIdx, M_csrColIdx.data(), M_csrNNZ * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(dM_csrValues, M_csrValues.data(), M_csrNNZ * sizeof(float), cudaMemcpyHostToDevice));

        // device setup cusparse M
        cusparseSpMatDescr_t Mat;

        CHECK_CUSPARSE(cusparseCreateCsr(&Mat, ENCODING_SIZE, batchSize, M_csrNNZ,
                                         dM_csrRowoffsets, dM_csrColIdx, dM_csrValues,
                                         CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                         CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

        // device setup cusparse spgemM result 
        cusparseSpMatDescr_t spgemM;

        CHECK_CUSPARSE(cusparseCreateCsr(&spgemM, csrRowoffsetsLength - 1, batchSize, 0,
                                         NULL, NULL, NULL,
                                         CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                         CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

        // device spgemM computation
        cusparseSpGEMMDescr_t spgemmDesc;
        CHECK_CUSPARSE(cusparseSpGEMM_createDescr(&spgemmDesc));
        CHECK_CUSPARSE(cusparseSpGEMM_workEstimation(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                     &alpha, mat, Mat, &beta, spgemM,
                                                     CUDA_R_32F, CUSPARSE_SPGEMM_ALG1,
                                                     spgemmDesc, &bufferSize1, NULL));
        CHECK_CUDA(cudaMalloc((void**) &dBuffer1, bufferSize1));
        CHECK_CUSPARSE(cusparseSpGEMM_workEstimation(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                     &alpha, mat, Mat, &beta, spgemM,
                                                     CUDA_R_32F, CUSPARSE_SPGEMM_ALG1,
                                                     spgemmDesc, &bufferSize1, dBuffer1));
        CHECK_CUSPARSE(cusparseSpGEMM_compute(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                              &alpha, mat, Mat, &beta, spgemM,
                                              CUDA_R_32F, CUSPARSE_SPGEMM_ALG1,
                                              spgemmDesc, &bufferSize2, NULL));
        CHECK_CUDA(cudaMalloc((void**) &dBuffer2, bufferSize2));
        CHECK_CUSPARSE(cusparseSpGEMM_compute(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                              &alpha, mat, Mat, &beta, spgemM,
                                              CUDA_R_32F, CUSPARSE_SPGEMM_ALG1,
                                              spgemmDesc, &bufferSize2, dBuffer2));

        // device get spgemM result
        int64_t spgemM_rows, spgemM_cols, spgemM_nnz;
        CHECK_CUSPARSE(cusparseSpMatGetSize(spgemM, &spgemM_rows, &spgemM_cols, &spgemM_nnz));
        CHECK_CUDA(cudaMalloc((void**) &dspgemM_csrColIdx, spgemM_nnz * sizeof(int)));
        CHECK_CUDA(cudaMalloc((void**) &dspgemM_csrValues, spgemM_nnz * sizeof(float)));
        CHECK_CUSPARSE(cusparseCsrSetPointers(spgemM, dspgemM_csrRowoffsets, dspgemM_csrColIdx, dspgemM_csrValues));
        CHECK_CUSPARSE(cusparseSpGEMM_copy(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &alpha, mat, Mat, &beta, spgemM,
                                           CUDA_R_32F, CUSPARSE_SPGEMM_ALG1, spgemmDesc));

        // host setup spgemM result
        int* spgemM_csrRowoffsets = new int[csrRowoffsetsLength];
        int* spgemM_csrColIdx = new int[spgemM_nnz];
        float* spgemM_csrValues = new float[spgemM_nnz];
        CHECK_CUDA(cudaMemcpy(spgemM_csrRowoffsets, dspgemM_csrRowoffsets, csrRowoffsetsLength * sizeof(int), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(spgemM_csrColIdx, dspgemM_csrColIdx, spgemM_nnz * sizeof(int), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(spgemM_csrValues, dspgemM_csrValues, spgemM_nnz * sizeof(float), cudaMemcpyDeviceToHost));
        
        // host result max
        for (int s = 0; s < batchSize; ++s) {
            std::vector<int> rowIdx;
            std::vector<float> rowValues;
            for (int j = 0; j < spgemM_nnz; ++j) {
                if (spgemM_csrColIdx[j] == s) {
                    rowIdx.push_back(getRowIdx(spgemM_csrRowoffsets, csrRowoffsetsLength, j));
                    rowValues.push_back(spgemM_csrValues[j]);
                }
            }
            
            // need to create an idx vector because we can't sort rowIdx directly
            //std::sort(rowIdx.data(), rowIdx.data() + rowIdx.size(), [&](int i, int j) {return rowValues[i] > rowValues[j];});
            std::vector<int> idx(rowIdx.size());
            std::iota(idx.begin(), idx.end(), 0);
            std::sort(idx.begin(), idx.end(), [&](int i, int j) {return rowValues[i] > rowValues[j];});

            if (rowIdx.size() >= n) {
                for (int j = 0; j < n; ++j) {
                    result[(i + s) * n + j] = rowIdx[idx[j]];
                }
            }
        }

        // host memory deallocation
        delete[] spgemM_csrRowoffsets;
        delete[] spgemM_csrColIdx;
        delete[] spgemM_csrValues;

        // device destroy descriptors
        CHECK_CUSPARSE(cusparseDestroySpMat(Mat));
        CHECK_CUSPARSE(cusparseDestroySpMat(spgemM));

        // device memory deallocation
        CHECK_CUDA(cudaFree(dM_csrRowoffsets));
        CHECK_CUDA(cudaFree(dM_csrColIdx));
        CHECK_CUDA(cudaFree(dM_csrValues));
        CHECK_CUDA(cudaFree(dspgemM_csrColIdx));
        CHECK_CUDA(cudaFree(dspgemM_csrValues));

        if (verbose != 0 && (i + batchSize) % verbose == 0) {
            std::cout << "Searched " << i + batchSize << " spectra in total..." << std::endl;
        }
    }

    // device destroy descriptors
    CHECK_CUSPARSE(cusparseDestroySpMat(mat));
    CHECK_CUSPARSE(cusparseDestroy(handle));

    // device memory deallocation
    CHECK_CUDA(cudaFree(dm_csrRowoffsets));
    CHECK_CUDA(cudaFree(dm_csrColIdx));
    CHECK_CUDA(cudaFree(dm_csrValues));
    CHECK_CUDA(cudaFree(dspgemM_csrRowoffsets));
    CHECK_CUDA(cudaFree(dBuffer1));
    CHECK_CUDA(cudaFree(dBuffer2));

    // host memory deallocation
    delete[] csrValues;

    return result;
}

/// <summary>
/// Free memory after result has been marshalled.
/// </summary>
/// <param name="result">The result array.</param>
/// <returns>0</returns>
int releaseMemoryCuda(int* result) {

    delete[] result;
    return 0;
}

/// <summary>
/// Returns the square for a given value x.
/// </summary>
/// <param name="x">The value to be squared.</param>
/// <returns>Square of x.</returns>
float squared(float x) {
    return x * x;
}

/// <summary>
/// Returns the PDF for a given x for the normal distribution given by mu and sigma.
/// </summary>
/// <param name="x">The value for which the PDF should be calculated.</param>
/// <param name="mu">The mu of the normal distribution.</param>
/// <param name="sigma">The sigma of the normal distribution.</param>
/// <returns>The PDF at x for the normal distribution given by mu and sigma. If sigma = 0 it returns 1.</returns>
float normpdf(float x, float mu, float sigma) {
    if (sigma == 0.0) {
        return 1.0;
    }
    return (ONE_OVER_SQRT_PI / sigma) * exp(-0.5 * squared((x - mu) / sigma));
}

/// <summary>
/// Gets the row index of a specific position in the csr_column_indices array.
/// </summary>
/// <param name="csrRowoffsets">The csr_row_offsets array.</param>
/// <param name="csrRowoffsetsLength">Length (int) of the csr_row_offsets array.</param>
/// <param name="colIdxPos">The position (int) of the element in question of the csr_column_indices array.</param>
/// <returns>Associated row index (int) or throws an error if the row index could not be found.</returns>
int getRowIdx(int* csrRowoffsets, int csrRowoffsetsLength, int colIdxPos) {
    for (int i = 0; i < csrRowoffsetsLength - 1; ++i) {
        if (csrRowoffsets[i] == csrRowoffsets[i + 1]) {
            continue;
        }
        if (csrRowoffsets[i + 1] > colIdxPos) {
            return i;
        }
    }

    throw std::logic_error("Couldn't find row index.");

    return -1;
}

BOOL APIENTRY DllMain( HMODULE hModule,
                       DWORD  ul_reason_for_call,
                       LPVOID lpReserved
                     )
{
    switch (ul_reason_for_call)
    {
    case DLL_PROCESS_ATTACH:
    case DLL_THREAD_ATTACH:
    case DLL_THREAD_DETACH:
    case DLL_PROCESS_DETACH:
        break;
    }
    return TRUE;
}

