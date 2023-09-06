// dllmain.cpp : Defines the entry point for the DLL application.
#include "pch.h"
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <numeric>
#include <algorithm>
#include <iostream>

const int versionMajor = 1;
const int versionMinor = 5;
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
const int APPROX_NNZ_PER_ROW = 100;
const double ONE_OVER_SQRT_PI = 0.39894228040143267793994605993438;

extern "C" {
    EXPORT int* findTopCandidates(int*, int*, 
                                  int*, int*, 
                                  int, int, 
                                  int, int, 
                                  int, float,
                                  bool, bool,
                                  int, int);

    EXPORT int* findTopCandidates2(int*, int*,
                                   int*, int*,
                                   int, int,
                                   int, int,
                                   int, float,
                                   bool, bool,
                                   int, int);

    EXPORT int* findTopCandidatesBatched(int*, int*,
                                         int*, int*,
                                         int, int,
                                         int, int,
                                         int, float,
                                         bool, bool,
                                         int,
                                         int, int);

    EXPORT int* findTopCandidatesBatched2(int*, int*,
                                          int*, int*,
                                          int, int,
                                          int, int,
                                          int, float,
                                          bool, bool,
                                          int,
                                          int, int);

    EXPORT int releaseMemory(int*);
}

float squared(float);
float normpdf(float, float, float);

/// <summary>
/// A function that calculates the top n candidates for each spectrum (SpM*SpV). 
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
/// <param name="normalize">If candidate vectors should be normalized to sum(elements) = 1 (bool).</param>
/// <param name="gaussianTol">If spectrum peaks should be modelled as normal distributions or not (bool).</param>
/// <param name="cores">Number of cores (int) used by Eigen.</param>
/// <param name="verbose">Print info every (int) processed spectra.</param>
/// <returns>An integer array of length sILength * n containing the indexes of the top n candidates for each spectrum.</returns>
int* findTopCandidates(int* candidatesValues, int* candidatesIdx, 
                       int* spectraValues, int* spectraIdx, 
                       int cVLength, int cILength, 
                       int sVLength, int sILength,
                       int n, float tolerance,
                       bool normalize, bool gaussianTol,
                       int cores, int verbose) {

    int usedCores = 0;
    Eigen::setNbThreads(cores);
    usedCores = Eigen::nbThreads();

    std::cout << "Running Eigen sparse vector search version " << versionMajor << "." << versionMinor << "." << versionFix << std::endl;
    std::cout << "Using " << usedCores << " threads in total." << std::endl;

    auto* m = new Eigen::SparseMatrix<float, Eigen::RowMajor>(cILength, ENCODING_SIZE);
    m->reserve(Eigen::VectorXi::Constant(cILength, APPROX_NNZ_PER_ROW));

    int currentRow = 0;
    for (int i = 0; i < cILength; ++i) {
        int startIter = candidatesIdx[i];
        int endIter = i + 1 == cILength ? cVLength : candidatesIdx[i + 1];
        int nrNonZero = endIter - startIter;
        float val = normalize ? 1.0 / (float) nrNonZero : 1.0;
        for (int j = startIter; j < endIter; ++j) {
            m->insert(currentRow, candidatesValues[j]) = val;
        }
        ++currentRow;
    }

    m->makeCompressed();

    auto* result = new int[sILength * n];
    float t = round(tolerance * MASS_MULTIPLIER);

    for (int i = 0; i < sILength; ++i) {
        int startIter = spectraIdx[i];
        int endIter = i + 1 == sILength ? sVLength : spectraIdx[i + 1];
        auto* v = new Eigen::SparseVector<float, Eigen::ColMajor>(ENCODING_SIZE);
        v->reserve(APPROX_NNZ_PER_ROW);
        for (int j = startIter; j < endIter; ++j) {
            auto currentPeak = spectraValues[j];
            auto minPeak = currentPeak - t > 0 ? currentPeak - t : 0;
            auto maxPeak = currentPeak + t < ENCODING_SIZE ? currentPeak + t : ENCODING_SIZE - 1;

            for (int k = minPeak; k <= maxPeak; ++k) {
                float currentVal = v->coeffRef(k);
                float newVal = gaussianTol ? normpdf((float) k, (float) currentPeak, (float) (t / 3.0)) : 1.0;
                v->coeffRef(k) = max(currentVal, newVal);
            }
        }
        
        auto* spmv = new Eigen::Vector<float, Eigen::Dynamic>(cILength);
        *spmv = Eigen::Product(*m, *v);

        std::vector<float> colValues;
        colValues.resize(spmv->size());
        Eigen::Map<Eigen::Vector<float, Eigen::Dynamic>>(colValues.data(), spmv->size()) = *spmv;

        auto* idx = new int[cILength];
        std::iota(idx, idx + cILength, 0);
        std::sort(idx, idx + cILength, [&](int i, int j) {return colValues[i] > colValues[j];});

        for (int j = 0; j < n; ++j) {
            result[i * n + j] = idx[j];
        }

        delete[] idx;

        spmv->resize(0);
        v->resize(0);
        delete spmv;
        delete v;
        spmv = NULL;
        v = NULL;

        if (verbose != 0 && (i + 1) % verbose == 0) {
            std::cout << "Searched " << i + 1 << " spectra in total..." << std::endl;
        }
    }

    m->resize(0, 0);
    delete m;
    m = NULL;

    return result;
}

/// <summary>
/// A function that calculates the top n candidates for each spectrum (SpM*V). 
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
/// <param name="normalize">If candidate vectors should be normalized to sum(elements) = 1 (bool).</param>
/// <param name="gaussianTol">If spectrum peaks should be modelled as normal distributions or not (bool).</param>
/// <param name="cores">Number of cores (int) used by Eigen.</param>
/// <param name="verbose">Print info every (int) processed spectra.</param>
/// <returns>An integer array of length sILength * n containing the indexes of the top n candidates for each spectrum.</returns>
int* findTopCandidates2(int* candidatesValues, int* candidatesIdx,
                        int* spectraValues, int* spectraIdx,
                        int cVLength, int cILength,
                        int sVLength, int sILength,
                        int n, float tolerance,
                        bool normalize, bool gaussianTol,
                        int cores, int verbose) {

    int usedCores = 0;
    Eigen::setNbThreads(cores);
    usedCores = Eigen::nbThreads();

    std::cout << "Running Eigen dense vector search version " << versionMajor << "." << versionMinor << "." << versionFix << std::endl;
    std::cout << "Using " << usedCores << " threads in total." << std::endl;

    auto* m = new Eigen::SparseMatrix<float, Eigen::RowMajor>(cILength, ENCODING_SIZE);
    m->reserve(Eigen::VectorXi::Constant(cILength, APPROX_NNZ_PER_ROW));

    int currentRow = 0;
    for (int i = 0; i < cILength; ++i) {
        int startIter = candidatesIdx[i];
        int endIter = i + 1 == cILength ? cVLength : candidatesIdx[i + 1];
        int nrNonZero = endIter - startIter;
        float val = normalize ? 1.0 / (float) nrNonZero : 1.0;
        for (int j = startIter; j < endIter; ++j) {
            m->insert(currentRow, candidatesValues[j]) = val;
        }
        ++currentRow;
    }

    m->makeCompressed();

    auto* result = new int[sILength * n];
    float t = round(tolerance * MASS_MULTIPLIER);

    for (int i = 0; i < sILength; ++i) {
        int startIter = spectraIdx[i];
        int endIter = i + 1 == sILength ? sVLength : spectraIdx[i + 1];
        auto* v = new Eigen::Vector<float, Eigen::Dynamic>(ENCODING_SIZE);
        v->setZero();
        for (int j = startIter; j < endIter; ++j) {
            auto currentPeak = spectraValues[j];
            auto minPeak = currentPeak - t > 0 ? currentPeak - t : 0;
            auto maxPeak = currentPeak + t < ENCODING_SIZE ? currentPeak + t : ENCODING_SIZE - 1;

            for (int k = minPeak; k <= maxPeak; ++k) {
                float currentVal = v->coeffRef(k);
                float newVal = gaussianTol ? normpdf((float) k, (float) currentPeak, (float) (t / 3.0)) : 1.0;
                v->coeffRef(k) = max(currentVal, newVal);
            }
        }

        auto* spmv = new Eigen::Vector<float, Eigen::Dynamic>(cILength);
        *spmv = Eigen::Product(*m, *v);

        std::vector<float> colValues;
        colValues.resize(spmv->size());
        Eigen::Map<Eigen::Vector<float, Eigen::Dynamic>>(colValues.data(), spmv->size()) = *spmv;

        auto* idx = new int[cILength];
        std::iota(idx, idx + cILength, 0);
        std::sort(idx, idx + cILength, [&](int i, int j) {return colValues[i] > colValues[j];});

        for (int j = 0; j < n; ++j) {
            result[i * n + j] = idx[j];
        }

        delete[] idx;

        spmv->resize(0);
        v->resize(0);
        delete spmv;
        delete v;
        spmv = NULL;
        v = NULL;

        if (verbose != 0 && (i + 1) % verbose == 0) {
            std::cout << "Searched " << i + 1 << " spectra in total..." << std::endl;
        }
    }

    m->resize(0, 0);
    delete m;
    m = NULL;

    return result;
}

/// <summary>
/// A function that calculates the top n candidates for each spectrum (SpM*SpM).
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
/// <param name="normalize">If candidate vectors should be normalized to sum(elements) = 1 (bool).</param>
/// <param name="gaussianTol">If spectrum peaks should be modelled as normal distributions or not (bool).</param>
/// <param name="batchSize">How many spectra (int) should be searched at once.</param>
/// <param name="cores">Number of cores (int) used by Eigen.</param>
/// <param name="verbose">Print info every (int) processed spectra.</param>
/// <returns>An integer array of length sILength * n containing the indexes of the top n candidates for each spectrum.</returns>
int* findTopCandidatesBatched(int* candidatesValues, int* candidatesIdx,
                              int* spectraValues, int* spectraIdx,
                              int cVLength, int cILength,
                              int sVLength, int sILength,
                              int n, float tolerance,
                              bool normalize, bool gaussianTol,
                              int batchSize,
                              int cores, int verbose) {

    int usedCores = 0;
    Eigen::setNbThreads(cores);
    usedCores = Eigen::nbThreads();

    std::cout << "Running Eigen sparse matrix search version " << versionMajor << "." << versionMinor << "." << versionFix << std::endl;
    std::cout << "Using " << usedCores << " threads in total." << std::endl;

    auto* m = new Eigen::SparseMatrix<float, Eigen::RowMajor>(cILength, ENCODING_SIZE);
    m->reserve(Eigen::VectorXi::Constant(cILength, APPROX_NNZ_PER_ROW));

    int currentRow = 0;
    for (int i = 0; i < cILength; ++i) {
        int startIter = candidatesIdx[i];
        int endIter = i + 1 == cILength ? cVLength : candidatesIdx[i + 1];
        int nrNonZero = endIter - startIter;
        float val = normalize ? 1.0 / (float) nrNonZero : 1.0;
        for (int j = startIter; j < endIter; ++j) {
            m->insert(currentRow, candidatesValues[j]) = val;
        }
        ++currentRow;
    }

    m->makeCompressed();

    auto* result = new int[sILength * n];
    float t = round(tolerance * MASS_MULTIPLIER);

    for (int i = 0; i < sILength; i += batchSize) {

        auto* M = new Eigen::SparseMatrix<float, Eigen::RowMajor>(ENCODING_SIZE, batchSize);
        std::vector<Eigen::Triplet<float>> M_entries;
        M_entries.reserve(1000 * batchSize);

        for (int s = 0; s < batchSize; ++s) {

            if (i + s >= sILength) {
                break;
            }

            int startIter = spectraIdx[i + s];
            int endIter = i + s + 1 == sILength ? sVLength : spectraIdx[i + s + 1];
            auto* v = new float[ENCODING_SIZE] {0.0};
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
            for (int j = 0; j < ENCODING_SIZE; ++j) {
                if (v[j] != 0.0) {
                    M_entries.push_back(Eigen::Triplet<float>(j, s, v[j]));
                }
            }
            delete[] v;
        }

        M->setFromTriplets(M_entries.begin(), M_entries.end());
        M->makeCompressed();

        auto* spmM = new Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>(cILength, batchSize);
        *spmM = Eigen::Product(*m, *M);

        for (int s = 0; s < batchSize; ++s) {

            if (i + s >= sILength) {
                break;
            }

            std::vector<float> colValues;
            colValues.resize(spmM->rows());
            Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>(colValues.data(), spmM->rows(), 1) = spmM->col(s);

            auto* idx = new int[cILength];
            std::iota(idx, idx + cILength, 0);
            std::sort(idx, idx + cILength, [&](int i, int j) {return colValues[i] > colValues[j];});

            for (int j = 0; j < n; ++j) {
                result[(i + s) * n + j] = idx[j];
            }

            delete[] idx;
        }

        spmM->resize(0, 0);
        delete spmM;
        spmM = NULL;
        M->resize(0, 0);
        delete M;
        M = NULL;

        if (verbose != 0 && (i + batchSize) % verbose == 0) {
            std::cout << "Searched " << i + batchSize << " spectra in total..." << std::endl;
        }
    }

    m->resize(0, 0);
    delete m;
    m = NULL;

    return result;
}

/// <summary>
/// A function that calculates the top n candidates for each spectrum (SpM*M).
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
/// <param name="normalize">If candidate vectors should be normalized to sum(elements) = 1 (bool).</param>
/// <param name="gaussianTol">If spectrum peaks should be modelled as normal distributions or not (bool).</param>
/// <param name="batchSize">How many spectra (int) should be searched at once.</param>
/// <param name="cores">Number of cores (int) used by Eigen.</param>
/// <param name="verbose">Print info every (int) processed spectra.</param>
/// <returns>An integer array of length sILength * n containing the indexes of the top n candidates for each spectrum.</returns>
int* findTopCandidatesBatched2(int* candidatesValues, int* candidatesIdx,
                               int* spectraValues, int* spectraIdx,
                               int cVLength, int cILength,
                               int sVLength, int sILength,
                               int n, float tolerance,
                               bool normalize, bool gaussianTol,
                               int batchSize,
                               int cores, int verbose) {

    int usedCores = 0;
    Eigen::setNbThreads(cores);
    usedCores = Eigen::nbThreads();

    std::cout << "Running Eigen dense matrix search version " << versionMajor << "." << versionMinor << "." << versionFix << std::endl;
    std::cout << "Using " << usedCores << " threads in total." << std::endl;

    auto* m = new Eigen::SparseMatrix<float, Eigen::RowMajor>(cILength, ENCODING_SIZE);
    m->reserve(Eigen::VectorXi::Constant(cILength, APPROX_NNZ_PER_ROW));

    int currentRow = 0;
    for (int i = 0; i < cILength; ++i) {
        int startIter = candidatesIdx[i];
        int endIter = i + 1 == cILength ? cVLength : candidatesIdx[i + 1];
        int nrNonZero = endIter - startIter;
        float val = normalize ? 1.0 / (float) nrNonZero : 1.0;
        for (int j = startIter; j < endIter; ++j) {
            m->insert(currentRow, candidatesValues[j]) = val;
        }
        ++currentRow;
    }

    m->makeCompressed();

    auto* result = new int[sILength * n];
    float t = round(tolerance * MASS_MULTIPLIER);

    for (int i = 0; i < sILength; i += batchSize) {

        auto* M = new Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>(ENCODING_SIZE, batchSize);
        M->setZero();

        for (int s = 0; s < batchSize; ++s) {

            if (i + s >= sILength) {
                break;
            }

            int startIter = spectraIdx[i + s];
            int endIter = i + s + 1 == sILength ? sVLength : spectraIdx[i + s + 1];
            for (int j = startIter; j < endIter; ++j) {
                auto currentPeak = spectraValues[j];
                auto minPeak = currentPeak - t > 0 ? currentPeak - t : 0;
                auto maxPeak = currentPeak + t < ENCODING_SIZE ? currentPeak + t : ENCODING_SIZE - 1;

                for (int k = minPeak; k <= maxPeak; ++k) {
                    float currentVal = M->coeff(k, s);
                    float newVal = gaussianTol ? normpdf((float) k, (float) currentPeak, (float) (t / 3.0)) : 1.0;
                    M->coeffRef(k, s) = max(currentVal, newVal);
                }
            }
        }

        auto* spmM = new Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>(cILength, batchSize);
        *spmM = Eigen::Product(*m, *M);

        for (int s = 0; s < batchSize; ++s) {

            if (i + s >= sILength) {
                break;
            }

            std::vector<float> colValues;
            colValues.resize(spmM->rows());
            Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>(colValues.data(), spmM->rows(), 1) = spmM->col(s);

            auto* idx = new int[cILength];
            std::iota(idx, idx + cILength, 0);
            std::sort(idx, idx + cILength, [&](int i, int j) {return colValues[i] > colValues[j];});

            for (int j = 0; j < n; ++j) {
                result[(i + s) * n + j] = idx[j];
            }

            delete[] idx;
        }

        spmM->resize(0, 0);
        delete spmM;
        spmM = NULL;
        M->resize(0, 0);
        delete M;
        M = NULL;

        if (verbose != 0 && (i + batchSize) % verbose == 0) {
            std::cout << "Searched " << i + batchSize << " spectra in total..." << std::endl;
        }
    }

    m->resize(0, 0);
    delete m;
    m = NULL;

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

