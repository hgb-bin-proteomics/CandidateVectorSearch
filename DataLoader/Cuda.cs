using System.Diagnostics;
using System.Runtime.InteropServices;

namespace FHOOE_IMP.MS_Annika.Utils.NonCleavableSearch
{
    public partial class DataLoader
    {
        const string dllCuda = @"VectorSearchCuda.dll";
        [DllImport(dllCuda, CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr findTopCandidatesCuda(IntPtr cR, IntPtr cI,
                                                           IntPtr sV, IntPtr sI,
                                                           int cRL, int cNNZ,
                                                           int sVL, int sIL,
                                                           int n, float tolerance,
                                                           bool normalize, bool gaussianTol,
                                                           int verbose);

        [DllImport(dllCuda, CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr findTopCandidatesCudaBatched(IntPtr cR, IntPtr cI,
                                                                  IntPtr sV, IntPtr sI,
                                                                  int cRL, int cNNZ,
                                                                  int sVL, int sIL,
                                                                  int n, float tolerance,
                                                                  bool normalize, bool gaussianTol,
                                                                  int batchSize,
                                                                  int verbose);

        [DllImport(dllCuda, CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr findTopCandidatesCudaBatched2(IntPtr cR, IntPtr cI,
                                                                   IntPtr sV, IntPtr sI,
                                                                   int cRL, int cNNZ,
                                                                   int sVL, int sIL,
                                                                   int n, float tolerance,
                                                                   bool normalize, bool gaussianTol,
                                                                   int batchSize,
                                                                   int verbose);

        [DllImport(dllCuda, CallingConvention = CallingConvention.Cdecl)]
        private static extern int releaseMemoryCuda(IntPtr result);

        public static int Cuda(int nrCandidates, int nrSpectra, int topN, Random r, bool batched, int batchMode)
        {
            // generate candidate vectors
            var csrRowoffsets = new int[nrCandidates + 1];
            var csrIdx = new int[nrCandidates * 100];
            var currentIdx = 0;
            for (int i = 0; i < csrIdx.Length; i += 100)
            {
                csrRowoffsets[currentIdx] = i;
                var tmpIdx = new int[100];
                for (int j = 0; j < tmpIdx.Length; j++)
                {
                    var val = r.Next(ENCODING_SIZE);
                    while (Array.Exists(tmpIdx, x => x == val))
                    {
                        val = r.Next(ENCODING_SIZE);
                    }
                    tmpIdx[j] = val;
                }
                Array.Sort(tmpIdx);
                for (int j = 0; j < tmpIdx.Length; j++)
                {
                    csrIdx[i + j] = tmpIdx[j];
                }
                currentIdx++;
                if (currentIdx % 5000 == 0)
                {
                    Console.WriteLine($"Generated {currentIdx} candidates...");
                }
            }
            // add the end of matrix as specified in CSR format
            csrRowoffsets[currentIdx++] = nrCandidates * 100;

            // generate spectra vectors
            var spectraValues = new int[nrSpectra * 500];
            var spectraIdx = new int[nrSpectra];
            currentIdx = 0;
            for (int i = 0; i < spectraValues.Length; i += 500)
            {
                spectraIdx[currentIdx] = i;
                var tmpValues = new int[500];
                for (int j = 0; j < tmpValues.Length; j++)
                {
                    var val = r.Next(ENCODING_SIZE);
                    while (Array.Exists(tmpValues, x => x == val))
                    {
                        val = r.Next(ENCODING_SIZE);
                    }
                    tmpValues[j] = val;
                }
                Array.Sort(tmpValues);
                for (int j = 0; j < tmpValues.Length; j++)
                {
                    spectraValues[i + j] = tmpValues[j];
                }
                currentIdx++;
            }

            // time c++ call
            var sw = Stopwatch.StartNew();

            // get pointer addresses and call c++ function
            var csrRowoffsetsLoc = GCHandle.Alloc(csrRowoffsets, GCHandleType.Pinned);
            var csrIdxLoc = GCHandle.Alloc(csrIdx, GCHandleType.Pinned);
            var sValuesLoc = GCHandle.Alloc(spectraValues, GCHandleType.Pinned);
            var sIdxLoc = GCHandle.Alloc(spectraIdx, GCHandleType.Pinned);
            var resultArray = new int[spectraIdx.Length * topN];
            var memStat = 1;
            try
            {
                if (!batched)
                {
                    IntPtr csrRowoffsetsPtr = csrRowoffsetsLoc.AddrOfPinnedObject();
                    IntPtr csrIdxPtr = csrIdxLoc.AddrOfPinnedObject();
                    IntPtr sValuesPtr = sValuesLoc.AddrOfPinnedObject();
                    IntPtr sIdxPtr = sIdxLoc.AddrOfPinnedObject();

                    IntPtr result = findTopCandidatesCuda(csrRowoffsetsPtr, csrIdxPtr,
                                                          sValuesPtr, sIdxPtr,
                                                          csrRowoffsets.Length, csrIdx.Length,
                                                          spectraValues.Length, spectraIdx.Length,
                                                          topN, (float) 0.02,
                                                          NORMALIZE, USE_GAUSSIAN,
                                                          1000);

                    Marshal.Copy(result, resultArray, 0, spectraIdx.Length * topN);

                    memStat = releaseMemoryCuda(result);
                }
                else
                {
                    if (batchMode == 2)
                    {
                        IntPtr csrRowoffsetsPtr = csrRowoffsetsLoc.AddrOfPinnedObject();
                        IntPtr csrIdxPtr = csrIdxLoc.AddrOfPinnedObject();
                        IntPtr sValuesPtr = sValuesLoc.AddrOfPinnedObject();
                        IntPtr sIdxPtr = sIdxLoc.AddrOfPinnedObject();

                        IntPtr result = findTopCandidatesCudaBatched2(csrRowoffsetsPtr, csrIdxPtr,
                                                                      sValuesPtr, sIdxPtr,
                                                                      csrRowoffsets.Length, csrIdx.Length,
                                                                      spectraValues.Length, spectraIdx.Length,
                                                                      topN, (float) 0.02,
                                                                      NORMALIZE, USE_GAUSSIAN,
                                                                      BATCH_SIZE,
                                                                      1000);

                        Marshal.Copy(result, resultArray, 0, spectraIdx.Length * topN);

                        memStat = releaseMemoryCuda(result);
                    }
                    else
                    {
                        IntPtr csrRowoffsetsPtr = csrRowoffsetsLoc.AddrOfPinnedObject();
                        IntPtr csrIdxPtr = csrIdxLoc.AddrOfPinnedObject();
                        IntPtr sValuesPtr = sValuesLoc.AddrOfPinnedObject();
                        IntPtr sIdxPtr = sIdxLoc.AddrOfPinnedObject();

                        IntPtr result = findTopCandidatesCudaBatched(csrRowoffsetsPtr, csrIdxPtr,
                                                                     sValuesPtr, sIdxPtr,
                                                                     csrRowoffsets.Length, csrIdx.Length,
                                                                     spectraValues.Length, spectraIdx.Length,
                                                                     topN, (float) 0.02,
                                                                     NORMALIZE, USE_GAUSSIAN,
                                                                     BATCH_SIZE,
                                                                     1000);

                        Marshal.Copy(result, resultArray, 0, spectraIdx.Length * topN);

                        memStat = releaseMemoryCuda(result);
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine("Something went wrong:");
                Console.WriteLine(ex.ToString());
            }
            finally
            {
                if (csrRowoffsetsLoc.IsAllocated) { csrRowoffsetsLoc.Free(); }
                if (csrIdxLoc.IsAllocated) { csrIdxLoc.Free(); }
                if (sValuesLoc.IsAllocated) { sValuesLoc.Free(); }
                if (sIdxLoc.IsAllocated) { sIdxLoc.Free(); }
            }

            // end time c++ call
            sw.Stop();

            for (int i = 0; i < topN; i++)
            {
                Console.WriteLine(resultArray[i]);
            }

            Console.WriteLine($"MemStat: {memStat}");
            var mode = batched ? batchMode == 2 ? "(SpMM)" : "(SpGEMM)" : "(SpMV)";
            Console.WriteLine($"Time for candidate search {mode}:");
            Console.WriteLine(sw.Elapsed.TotalSeconds.ToString());

            //
            GC.Collect();
            GC.WaitForPendingFinalizers();

            return 0;
        }
    }
}
