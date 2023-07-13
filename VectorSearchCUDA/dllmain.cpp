// dllmain.cpp : Defines the entry point for the DLL application.
#include "pch.h"
#include <cuda_runtime_api.h>
#include <cusparse.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <numeric>
#include <algorithm>

const int versionMajor = 1;
const int versionMinor = 1;
const int versionFix = 0;

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
                                      int, float);

    EXPORT int releaseMemoryCuda(int*);
}

float squared(float);
float normpdf(float, float, float);

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
/// <returns>An integer array of length sILength * n containing the indexes of the top n candidates for each spectrum.</returns>
int* findTopCandidatesCuda(int* csrRowoffsets, int* csrColIdx, 
                           int* spectraValues, int* spectraIdx,
                           int csrRowoffsetsLength, int csrNNZ,
                           int sVLength, int sILength,
                           int n, float tolerance) {

    std::cout << "Running CUDA vector search version " << versionMajor << "." << versionMinor << "." << versionFix << std::endl;

    int t = round(tolerance * MASS_MULTIPLIER);
    int* result = new int[sILength * n];
    float* csrValues = new float[csrNNZ];
    float* MVresult = new float[csrRowoffsetsLength - 1] {0.0};

    // create csrValues
    for (int i = 0; i < csrRowoffsetsLength - 1; ++i) {
        int startIter = csrRowoffsets[i];
        int endIter = csrRowoffsets[i + 1];
        int nrNonZero = endIter - startIter;
        float val = 1.0 / nrNonZero;
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
                float newVal = normpdf((float) k, (float) currentPeak, (float) t / 3);
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

