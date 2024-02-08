using System.Runtime.InteropServices;

namespace VectorSearchInterface
{
    public class VectorSearchAPI
    {
        const string dllCPU = @"VectorSearch.dll";
        const string dllGPU = @"VectorSearchCUDA.dll";

        #region CPU_Methods

        /// <summary>
        /// Enum of available computation methods on the CPU:
        /// - i32CPU_DV: Sparse matrix - dense vector multiplication using integer operations.
        /// - f32CPU_DV: Sparse matrix - dense vector multiplication using float operations.
        /// - i32CPU_DM: Sparse matrix - dense matrix multiplication using integer operations.
        /// - f32CPU_DM: Sparse matrix - dense matrix multiplication using float operations.
        /// - i32CPU_SV: Sparse matrix - sparse vector multiplication using integer operations.
        /// - f32CPU_SV: Sparse matrix - sparse vector multiplication using float operations.
        /// - i32CPU_SM: Sparse matrix - sparse matrix multiplication using integer operations.
        /// - f32CPU_SM: Sparse matrix - sparse matrix multiplication using float operations.
        /// </summary>
        public enum CPU_METHODS
        {
            i32CPU_DV,
            f32CPU_DV,
            i32CPU_DM,
            f32CPU_DM,
            i32CPU_SV,
            f32CPU_SV,
            i32CPU_SM,
            f32CPU_SM
        }

        #endregion

        #region GPU_Methods

        /// <summary>
        /// Enum of available computation methods on the GPU:
        /// - f32GPU_DV: Sparse matrix - dense vector multiplication using float operations.
        /// - f32GPU_DM: Sparse matrix - dense matrix multiplication using float operations.
        /// - f32GPU_SM: Sparse matrix - sparse matrix multiplication using float operations.
        /// </summary>
        public enum GPU_METHODS
        {
            f32GPU_DV,
            f32GPU_DM,
            f32GPU_SM
        }

        #endregion

        #region VectorSearch.dll_import

        [DllImport(dllCPU, CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr findTopCandidates(IntPtr cV, IntPtr cI, IntPtr sV, IntPtr sI,
                                                       int cVL, int cIL, int sVL, int sIL,
                                                       int n, float tolerance,
                                                       bool normalize, bool gaussianTol,
                                                       int cores, int verbose);

        [DllImport(dllCPU, CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr findTopCandidatesInt(IntPtr cV, IntPtr cI, IntPtr sV, IntPtr sI,
                                                          int cVL, int cIL, int sVL, int sIL,
                                                          int n, float tolerance,
                                                          bool normalize, bool gaussianTol,
                                                          int cores, int verbose);

        [DllImport(dllCPU, CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr findTopCandidatesBatched(IntPtr cV, IntPtr cI, IntPtr sV, IntPtr sI,
                                                              int cVL, int cIL, int sVL, int sIL,
                                                              int n, float tolerance,
                                                              bool normalize, bool gaussianTol,
                                                              int batchSize,
                                                              int cores, int verbose);

        [DllImport(dllCPU, CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr findTopCandidatesBatchedInt(IntPtr cV, IntPtr cI, IntPtr sV, IntPtr sI,
                                                                 int cVL, int cIL, int sVL, int sIL,
                                                                 int n, float tolerance,
                                                                 bool normalize, bool gaussianTol,
                                                                 int batchSize,
                                                                 int cores, int verbose);

        [DllImport(dllCPU, CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr findTopCandidates2(IntPtr cV, IntPtr cI, IntPtr sV, IntPtr sI,
                                                        int cVL, int cIL, int sVL, int sIL,
                                                        int n, float tolerance,
                                                        bool normalize, bool gaussianTol,
                                                        int cores, int verbose);

        [DllImport(dllCPU, CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr findTopCandidates2Int(IntPtr cV, IntPtr cI, IntPtr sV, IntPtr sI,
                                                           int cVL, int cIL, int sVL, int sIL,
                                                           int n, float tolerance,
                                                           bool normalize, bool gaussianTol,
                                                           int cores, int verbose);

        [DllImport(dllCPU, CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr findTopCandidatesBatched2(IntPtr cV, IntPtr cI, IntPtr sV, IntPtr sI,
                                                               int cVL, int cIL, int sVL, int sIL,
                                                               int n, float tolerance,
                                                               bool normalize, bool gaussianTol,
                                                               int batchSize,
                                                               int cores, int verbose);

        [DllImport(dllCPU, CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr findTopCandidatesBatched2Int(IntPtr cV, IntPtr cI, IntPtr sV, IntPtr sI,
                                                                  int cVL, int cIL, int sVL, int sIL,
                                                                  int n, float tolerance,
                                                                  bool normalize, bool gaussianTol,
                                                                  int batchSize,
                                                                  int cores, int verbose);

        [DllImport(dllCPU, CallingConvention = CallingConvention.Cdecl)]
        private static extern int releaseMemory(IntPtr result);

        #endregion

        #region VectorSearchCUDA.dll_import

        [DllImport(dllGPU, CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr findTopCandidatesCuda(IntPtr cR, IntPtr cI, IntPtr sV, IntPtr sI,
                                                           int cRL, int cNNZ, int sVL, int sIL,
                                                           int n, float tolerance,
                                                           bool normalize, bool gaussianTol,
                                                           int verbose);

        [DllImport(dllGPU, CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr findTopCandidatesCudaBatched(IntPtr cR, IntPtr cI, IntPtr sV, IntPtr sI,
                                                                  int cRL, int cNNZ, int sVL, int sIL,
                                                                  int n, float tolerance,
                                                                  bool normalize, bool gaussianTol,
                                                                  int batchSize,
                                                                  int verbose);

        [DllImport(dllGPU, CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr findTopCandidatesCudaBatched2(IntPtr cR, IntPtr cI, IntPtr sV, IntPtr sI,
                                                                   int cRL, int cNNZ, int sVL, int sIL,
                                                                   int n, float tolerance,
                                                                   bool normalize, bool gaussianTol,
                                                                   int batchSize,
                                                                   int verbose);

        [DllImport(dllGPU, CallingConvention = CallingConvention.Cdecl)]
        private static extern int releaseMemoryCuda(IntPtr result);

        #endregion

        #region CPU_search

        /// <summary>
        /// Calculates the top n candidates for each spectrum on the CPU using Eigen.
        /// </summary>
        /// <param name="candidatesValues">An integer array of theoretical ion m/z values for all candidates flattened.</param>
        /// <param name="candidatesIdx">An integer array that contains indices indicating where each candidate starts in candidatesValues.</param>
        /// <param name="spectraValues">An integer array of peak m/z values from experimental spectra flattened.</param>
        /// <param name="spectraIdx">An integer array that contains indices indicating where each spectrum starts in spectraValues.</param>
        /// <param name="topN">The number (int) of top candidates that should be returned for each spectrum.</param>
        /// <param name="tolerance">Tolerance used for matching peaks in Dalton (float).</param>
        /// <param name="normalize">Whether or not the candidate scores should be normalized by candidate length (bool).</param>
        /// <param name="useGaussianTol">Whether or not experimental peaks should be modelled as gaussian normal distributions (bool).</param>
        /// <param name="batchSize">If a batched approach is used, how big should batches be (integer).</param>
        /// <param name="method">Which matrix multiplication method should be used. See enum CPU_METHODS.</param>
        /// <param name="cores">The number of CPU cores that should be used for computation (int).</param>
        /// <param name="verbose">An integer parameter controlling how often progress should be printed to std::out. If 0 no progress will be printed.</param>
        /// <param name="memStat">An integer out parameter indicating if memory was successfully freed after execution, 0 = success, 1 = error.</param>
        /// <returns>An integer array with length (number of spectra * topN) containing the indices of the top n candidates for every spectrum.</returns>
        public static int[] searchCPU(ref int[] candidatesValues, ref int[] candidatesIdx, ref int[] spectraValues, ref int[] spectraIdx,
                                      int topN, float tolerance, bool normalize, bool useGaussianTol,
                                      int batchSize, CPU_METHODS method, int cores, int verbose,
                                      out int memStat)
        {
            var cValuesLoc = GCHandle.Alloc(candidatesValues, GCHandleType.Pinned);
            var cIdxLoc = GCHandle.Alloc(candidatesIdx, GCHandleType.Pinned);
            var sValuesLoc = GCHandle.Alloc(spectraValues, GCHandleType.Pinned);
            var sIdxLoc = GCHandle.Alloc(spectraIdx, GCHandleType.Pinned);

            int cVLength = candidatesValues.Length;
            int cILength = candidatesIdx.Length;
            int sVLength = spectraValues.Length;
            int sILength = spectraIdx.Length;

            var resultArray = new int[sILength * topN];

            memStat = 1;

            try
            {
                IntPtr cValuesPtr = cValuesLoc.AddrOfPinnedObject();
                IntPtr cIdxPtr = cIdxLoc.AddrOfPinnedObject();
                IntPtr sValuesPtr = sValuesLoc.AddrOfPinnedObject();
                IntPtr sIdxPtr = sIdxLoc.AddrOfPinnedObject();

                switch(method)
                {
                    case CPU_METHODS.f32CPU_SV:
                        IntPtr result1 = findTopCandidates(cValuesPtr, cIdxPtr, sValuesPtr, sIdxPtr,
                                                           cVLength, cILength, sVLength, sILength,
                                                           topN, tolerance, normalize, useGaussianTol,
                                                           cores, verbose);

                        Marshal.Copy(result1, resultArray, 0, sILength * topN);

                        memStat = releaseMemory(result1);
                        break;

                    case CPU_METHODS.f32CPU_DV:
                        IntPtr result2 = findTopCandidates2(cValuesPtr, cIdxPtr, sValuesPtr, sIdxPtr,
                                                            cVLength, cILength, sVLength, sILength,
                                                            topN, tolerance, normalize, useGaussianTol,
                                                            cores, verbose);

                        Marshal.Copy(result2, resultArray, 0, sILength * topN);

                        memStat = releaseMemory(result2);
                        break;

                    case CPU_METHODS.f32CPU_SM:
                        IntPtr result3 = findTopCandidatesBatched(cValuesPtr, cIdxPtr, sValuesPtr, sIdxPtr,
                                                                  cVLength, cILength, sVLength, sILength,
                                                                  topN, tolerance, normalize, useGaussianTol, batchSize,
                                                                  cores, verbose);

                        Marshal.Copy(result3, resultArray, 0, sILength * topN);

                        memStat = releaseMemory(result3);
                        break;

                    case CPU_METHODS.f32CPU_DM:
                        IntPtr result4 = findTopCandidatesBatched2(cValuesPtr, cIdxPtr, sValuesPtr, sIdxPtr,
                                                                   cVLength, cILength, sVLength, sILength,
                                                                   topN, tolerance, normalize, useGaussianTol, batchSize,
                                                                   cores, verbose);

                        Marshal.Copy(result4, resultArray, 0, sILength * topN);

                        memStat = releaseMemory(result4);
                        break;

                    case CPU_METHODS.i32CPU_DM:
                        IntPtr result5 = findTopCandidatesBatched2Int(cValuesPtr, cIdxPtr, sValuesPtr, sIdxPtr,
                                                                      cVLength, cILength, sVLength, sILength,
                                                                      topN, tolerance, normalize, useGaussianTol, batchSize,
                                                                      cores, verbose);

                        Marshal.Copy(result5, resultArray, 0, sILength * topN);

                        memStat = releaseMemory(result5);
                        break;

                    case CPU_METHODS.i32CPU_DV:
                        IntPtr result6 = findTopCandidates2Int(cValuesPtr, cIdxPtr, sValuesPtr, sIdxPtr,
                                                               cVLength, cILength, sVLength, sILength,
                                                               topN, tolerance, normalize, useGaussianTol,
                                                               cores, verbose);

                        Marshal.Copy(result6, resultArray, 0, sILength * topN);

                        memStat = releaseMemory(result6);
                        break;

                    case CPU_METHODS.i32CPU_SV:
                        IntPtr result7 = findTopCandidatesInt(cValuesPtr, cIdxPtr, sValuesPtr, sIdxPtr,
                                                              cVLength, cILength, sVLength, sILength,
                                                              topN, tolerance, normalize, useGaussianTol,
                                                              cores, verbose);

                        Marshal.Copy(result7, resultArray, 0, sILength * topN);

                        memStat = releaseMemory(result7);
                        break;

                    default:
                        IntPtr result = findTopCandidatesBatchedInt(cValuesPtr, cIdxPtr, sValuesPtr, sIdxPtr,
                                                                    cVLength, cILength, sVLength, sILength,
                                                                    topN, tolerance, normalize, useGaussianTol, batchSize,
                                                                    cores, verbose);

                        Marshal.Copy(result, resultArray, 0, sILength * topN);

                        memStat = releaseMemory(result);
                        break;
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine("Something went wrong:");
                Console.WriteLine(ex.ToString());
                memStat = 1;
            }
            finally
            {
                if (cValuesLoc.IsAllocated) { cValuesLoc.Free(); }
                if (cIdxLoc.IsAllocated) { cIdxLoc.Free(); }
                if (sValuesLoc.IsAllocated) { sValuesLoc.Free(); }
                if (sIdxLoc.IsAllocated) { sIdxLoc.Free(); }
            }

            return resultArray;
        }

        #endregion

        #region GPU_search

        /// <summary>
        /// Calculates the top n candidates for each spectrum on the (Nvidia) GPU using CUDA.
        /// </summary>
        /// <param name="csrRowoffsets">An integer array of rowoffsets of the CSR sparse matrix with length (rows + 1 = number of candidates + 1).</param>
        /// <param name="csrColIdx">An integer array of column indices of the CSR sparse matrix with length (NNZ = total number of theoretical ions).</param>
        /// <param name="spectraValues">An integer array of peak m/z values from experimental spectra flattened.</param>
        /// <param name="spectraIdx">An integer array that contains indices indicating where each spectrum starts in spectraValues.</param>
        /// <param name="topN">The number (int) of top candidates that should be returned for each spectrum.</param>
        /// <param name="tolerance">Tolerance used for matching peaks in Dalton (float).</param>
        /// <param name="normalize">Whether or not the candidate scores should be normalized by candidate length (bool).</param>
        /// <param name="useGaussianTol">Whether or not experimental peaks should be modelled as gaussian normal distributions (bool).</param>
        /// <param name="batchSize">If a batched approach is used, how big should batches be (integer).</param>
        /// <param name="method">Which matrix multiplication method should be used. See enum GPU_METHODS.</param>
        /// <param name="verbose">An integer parameter controlling how often progress should be printed to std::out. If 0 no progress will be printed.</param>
        /// <param name="memStat">An integer out parameter indicating if memory was successfully freed after execution, 0 = success, 1 = error.</param>
        /// <returns>An integer array with length (number of spectra * topN) containing the indices of the top n candidates for every spectrum.</returns>
        public static int[] searchGPU(ref int[] csrRowoffsets, ref int[] csrColIdx, ref int[] spectraValues, ref int[] spectraIdx,
                                      int topN, float tolerance, bool normalize, bool useGaussianTol,
                                      int batchSize, GPU_METHODS method, int verbose,
                                      out int memStat)
        {
            var csrRowoffsetsLoc = GCHandle.Alloc(csrRowoffsets, GCHandleType.Pinned);
            var csrIdxLoc = GCHandle.Alloc(csrColIdx, GCHandleType.Pinned);
            var sValuesLoc = GCHandle.Alloc(spectraValues, GCHandleType.Pinned);
            var sIdxLoc = GCHandle.Alloc(spectraIdx, GCHandleType.Pinned);

            int cRLength = csrRowoffsets.Length;
            int cILength = csrColIdx.Length;
            int sVLength = spectraValues.Length;
            int sILength = spectraIdx.Length;

            var resultArray = new int[sILength * topN];

            memStat = 1;

            try
            {
                IntPtr csrRowoffsetsPtr = csrRowoffsetsLoc.AddrOfPinnedObject();
                IntPtr csrIdxPtr = csrIdxLoc.AddrOfPinnedObject();
                IntPtr sValuesPtr = sValuesLoc.AddrOfPinnedObject();
                IntPtr sIdxPtr = sIdxLoc.AddrOfPinnedObject();

                switch(method)
                {
                    case GPU_METHODS.f32GPU_DM:
                        IntPtr result1 = findTopCandidatesCudaBatched2(csrRowoffsetsPtr, csrIdxPtr, sValuesPtr, sIdxPtr,
                                                                       cRLength, cILength, sVLength, sILength,
                                                                       topN, tolerance, normalize, useGaussianTol, batchSize,
                                                                       verbose);

                        Marshal.Copy(result1, resultArray, 0, sILength * topN);

                        memStat = releaseMemoryCuda(result1);
                        break;

                    case GPU_METHODS.f32GPU_SM:
                        IntPtr result2 = findTopCandidatesCudaBatched(csrRowoffsetsPtr, csrIdxPtr, sValuesPtr, sIdxPtr,
                                                                      cRLength, cILength, sVLength, sILength,
                                                                      topN, tolerance, normalize, useGaussianTol, batchSize,
                                                                      verbose);

                        Marshal.Copy(result2, resultArray, 0, sILength * topN);

                        memStat = releaseMemoryCuda(result2);
                        break;

                    default:
                        IntPtr result = findTopCandidatesCuda(csrRowoffsetsPtr, csrIdxPtr, sValuesPtr, sIdxPtr,
                                                              cRLength, cILength, sVLength, sILength,
                                                              topN, tolerance, normalize, useGaussianTol,
                                                              verbose);

                        Marshal.Copy(result, resultArray, 0, sILength * topN);

                        memStat = releaseMemoryCuda(result);
                        break;
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine("Something went wrong:");
                Console.WriteLine(ex.ToString());
                memStat = 1;
            }
            finally
            {
                if (csrRowoffsetsLoc.IsAllocated) { csrRowoffsetsLoc.Free(); }
                if (csrIdxLoc.IsAllocated) { csrIdxLoc.Free(); }
                if (sValuesLoc.IsAllocated) { sValuesLoc.Free(); }
                if (sIdxLoc.IsAllocated) { sIdxLoc.Free(); }
            }

            return resultArray;
        }

        #endregion
    }
}