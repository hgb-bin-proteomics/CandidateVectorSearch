using System.Runtime.InteropServices;

namespace VectorSearchInterface
{
    public class VectorSearchAPI
    {
        const string dllCPU = @"VectorSearch.dll";
        const string dllGPU = @"VectorSearchCUDA.dll";

        #region VectorSearch.dll_import

        [DllImport(dllCPU, CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr findTopCandidates(IntPtr cV, IntPtr cI, IntPtr sV, IntPtr sI,
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
        private static extern IntPtr findTopCandidates2(IntPtr cV, IntPtr cI, IntPtr sV, IntPtr sI,
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

        public static int[] searchCPU(ref int[] candidatesValues, ref int[] candidatesIdx, ref int[] spectraValues, ref int[] spectraIdx,
                                      int topN, float tolerance, bool normalize, bool useGaussianTol,
                                      bool batched, int batchSize, bool useSparse, int cores, int verbose,
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

                if (!batched && useSparse)
                {
                    IntPtr result = findTopCandidates(cValuesPtr, cIdxPtr, sValuesPtr, sIdxPtr,
                                                      cVLength, cILength, sVLength, sILength,
                                                      topN, tolerance, normalize, useGaussianTol,
                                                      cores, verbose);

                    Marshal.Copy(result, resultArray, 0, sILength * topN);

                    memStat = releaseMemory(result);
                }
                else if (!batched && !useSparse)
                {
                    IntPtr result = findTopCandidates2(cValuesPtr, cIdxPtr, sValuesPtr, sIdxPtr,
                                                       cVLength, cILength, sVLength, sILength,
                                                       topN, tolerance, normalize, useGaussianTol,
                                                       cores, verbose);

                    Marshal.Copy(result, resultArray, 0, sILength * topN);

                    memStat = releaseMemory(result);
                }
                else if (batched && useSparse)
                {
                    IntPtr result = findTopCandidatesBatched(cValuesPtr, cIdxPtr, sValuesPtr, sIdxPtr,
                                                             cVLength, cILength, sVLength, sILength,
                                                             topN, tolerance, normalize, useGaussianTol, batchSize,
                                                             cores, verbose);

                    Marshal.Copy(result, resultArray, 0, sILength * topN);

                    memStat = releaseMemory(result);
                }
                else if (batched && !useSparse)
                {
                    IntPtr result = findTopCandidatesBatched2(cValuesPtr, cIdxPtr, sValuesPtr, sIdxPtr,
                                                              cVLength, cILength, sVLength, sILength,
                                                              topN, tolerance, normalize, useGaussianTol, batchSize,
                                                              cores, verbose);

                    Marshal.Copy(result, resultArray, 0, sILength * topN);

                    memStat = releaseMemory(result);
                }
                else
                {
                    Console.WriteLine("Impossible case!");
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

        public static int[] searchGPU(ref int[] csrRowoffsets, ref int[] csrColIdx, ref int[] spectraValues, ref int[] spectraIdx,
                                      int topN, float tolerance, bool normalize, bool useGaussianTol,
                                      bool batched, int batchSize, bool useSparse, int verbose,
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

                if (!batched)
                {
                    IntPtr result = findTopCandidatesCuda(csrRowoffsetsPtr, csrIdxPtr, sValuesPtr, sIdxPtr,
                                                          cRLength, cILength, sVLength, sILength,
                                                          topN, tolerance, normalize, useGaussianTol,
                                                          verbose);

                    Marshal.Copy(result, resultArray, 0, sILength * topN);

                    memStat = releaseMemoryCuda(result);
                }
                else
                {
                    if (!useSparse)
                    {
                        IntPtr result = findTopCandidatesCudaBatched2(csrRowoffsetsPtr, csrIdxPtr, sValuesPtr, sIdxPtr,
                                                                      cRLength, cILength, sVLength, sILength,
                                                                      topN, tolerance, normalize, useGaussianTol, batchSize,
                                                                      verbose);

                        Marshal.Copy(result, resultArray, 0, sILength * topN);

                        memStat = releaseMemoryCuda(result);
                    }
                    else
                    {


                        IntPtr result = findTopCandidatesCudaBatched(csrRowoffsetsPtr, csrIdxPtr, sValuesPtr, sIdxPtr,
                                                                     cRLength, cILength, sVLength, sILength,
                                                                     topN, tolerance, normalize, useGaussianTol, batchSize,
                                                                     verbose);

                        Marshal.Copy(result, resultArray, 0, sILength * topN);

                        memStat = releaseMemoryCuda(result);
                    }
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