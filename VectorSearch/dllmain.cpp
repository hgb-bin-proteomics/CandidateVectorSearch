// dllmain.cpp : Defines the entry point for the DLL application.
#include "pch.h"
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>

#define METHOD_EXPORTS
#ifdef METHOD_EXPORTS
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __declspec(dllimport)
#endif

const int MASS_RANGE = 1300;
const int MASS_MULTIPLIER = 100;
const int ENCODING_SIZE = MASS_RANGE * MASS_MULTIPLIER;
const int APPROX_NNZ_PER_ROW = 100;
const double ONE_OVER_SQRT_PI = 0.39894228040143267793994605993438;

extern "C" {
    EXPORT int* findTopCandidates(int*, int*, int*, int*, 
                                  int, int, int, int, 
                                  int, float);

    EXPORT int releaseMemory(int*);
}

float squared(float);
float normpdf(float, float, float);

/// <summary>
/// A function that calculates the top n candidates for each spectrum.
/// </summary>
/// <param name="candidatesValues">An integer array of theoretical ion masses for all candidates flattened.</param>
/// <param name="candidatesIdx">An integer array that contains indices of where each candidate starts in candidatesValues.</param>
/// <param name="spectraValues">An integer array of peaks from experimental spectra flattened.</param>
/// <param name="spectraIdx">An integer array that contains indices of where each spectrum starts in spectraValues.</param>
/// <param name="cVLength">Length (int) of candidatesValues.</param>
/// <param name="cILength">Length (int) of candidatesIdx.</param>
/// <param name="sVLength">Length (int) of spectraValues.</param>
/// <param name="sILength">Length (int) of spectraIdx.</param>
/// <param name="n">How many of the best hits should be returned (int).</param>
/// <param name="tolerance">Tolerance for peak matching (float).</param>
/// <returns>An integer array of length sILength * n containing the indexes of the top n candidates for each spectrum.</returns>
int* findTopCandidates(int* candidatesValues, int* candidatesIdx, int* spectraValues, int* spectraIdx, 
                       int cVLength, int cILength, int sVLength, int sILength,
                       int n, float tolerance) {

    auto* m = new Eigen::SparseMatrix<float, Eigen::RowMajor>(cILength, ENCODING_SIZE);
    m->reserve(Eigen::VectorXi::Constant(cILength, APPROX_NNZ_PER_ROW));

    int currentRow = 0;
    for (int i = 0; i < cILength; ++i) {
        int startIter = candidatesIdx[i];
        int endIter = i + 1 == cILength ? cVLength : candidatesIdx[i + 1];
        int nrNonZero = endIter - startIter;
        float val = 1.0 / nrNonZero;
        for (int j = startIter; j < endIter; ++j) {
            m->insert(currentRow, candidatesValues[j]) = val;
        }
        ++currentRow;
    }

    m->makeCompressed();

    //auto result = new std::vector<int>;
    //result.reserve(sILength * n);
    auto* result = new int[sILength * n];
    auto t = round(tolerance * MASS_MULTIPLIER);

    for (int i = 0; i < sILength; ++i) {
        int startIter = spectraIdx[i];
        int endIter = i + 1 == sILength ? sVLength : spectraIdx[i + 1];
        auto* v = new Eigen::SparseVector<float, Eigen::ColMajor>(ENCODING_SIZE);
        for (int j = startIter; j < endIter; ++j) {
            auto currentPeak = spectraValues[j];
            auto minPeak = currentPeak - t > 0 ? currentPeak - t : 0;
            auto maxPeak = currentPeak + t < ENCODING_SIZE ? currentPeak + t : ENCODING_SIZE - 1;

            for (int k = minPeak; k <= maxPeak; ++k) {
                float currentVal = v->coeffRef(k);
                float newVal = normpdf((float) k, (float) currentPeak, (float) t / 3);
                v->coeffRef(k) = max(currentVal, newVal);
            }
        }
        
        auto* spmv = new Eigen::Vector<float, Eigen::Dynamic>(cILength);
        *spmv = Eigen::Product(*m, *v);

        for (int j = 0; j < n; ++j) {
            Eigen::Index max_idx;
            float max = spmv->maxCoeff(&max_idx);
            //result.push_back((int) max_idx);
            result[i * n + j] = (int) max_idx;
            spmv->coeffRef(max_idx) = 0.0;
        }

        delete spmv;
        delete v;
    }

    delete m;

    //return result.data();
    return result;
}

/// <summary>
/// Free memory after result has been marshalled.
/// </summary>
/// <param name="result">The result array.</param>
/// <returns>0</returns>
int releaseMemory(int* result) {

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
/// <returns>The PDF at x for the normal distribution given by mu and sigma</returns>
float normpdf(float x, float mu, float sigma) {
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

