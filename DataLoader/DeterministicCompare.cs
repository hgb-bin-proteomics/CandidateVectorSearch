using System.Diagnostics;
using System.Runtime.InteropServices;

namespace FHOOE_IMP.MS_Annika.Utils.NonCleavableSearch
{
    public partial class DataLoader
    {
        public static int DeterministicCompare()
        {
            // EXAMPLE PROBLEM
            //
            // need to set encoding size to 5 in c++ dlls
            //
            // [ [0, 1, 0, 0, 1]        - 2
            //   [1, 1, 1, 1, 0]        - 4
            //   [0, 1, 1, 1, 0]        - 3
            //   [0, 1, 0, 0, 1]        - 2
            //   [1, 1, 1, 1, 1] ]      - 5
            //
            // [0, 1, 1, 0, 0]
            // [1, 1, 1, 0, 1]
            // 
            // Result 1:
            // [0.5, 0.5, 0.6, 0.5, 0.4]
            //
            // Result 2;
            // [1, 0.75, 0.6, 1, 0.8]
            // generate candidate vectors
            var candidateValues = new int[] { 1, 4, 0, 1, 2, 3, 1, 2, 3, 1, 4, 0, 1, 2, 3, 4};
            var candidatesIdx = new int[] { 0, 2, 6, 9, 11 };
            var csrRowoffsets = new int[] { 0, 2, 6, 9, 11, 16 };
            var csrIdx = new int[] { 1, 4, 0, 1, 2, 3, 1, 2, 3, 1, 4, 0, 1, 2, 3, 4 };

            // generate spectra vectors
            var spectraValues = new int[] { 1, 2, 0, 1, 2, 4 };
            var spectraIdx = new int[] { 0, 2 };

            // topN
            var topN = 5;

            // time c++ call
            var sw = Stopwatch.StartNew();

            // get pointer addresses and call c++ function
            var cValuesLoc = GCHandle.Alloc(candidateValues, GCHandleType.Pinned);
            var cIdxLoc = GCHandle.Alloc(candidatesIdx, GCHandleType.Pinned);
            var csrRowoffsetsLoc = GCHandle.Alloc(csrRowoffsets, GCHandleType.Pinned);
            var csrIdxLoc = GCHandle.Alloc(csrIdx, GCHandleType.Pinned);
            var sValuesLoc = GCHandle.Alloc(spectraValues, GCHandleType.Pinned);
            var sIdxLoc = GCHandle.Alloc(spectraIdx, GCHandleType.Pinned);
            var resultArrayEigen = new int[spectraIdx.Length * topN];
            var resultArrayCuda = new int[spectraIdx.Length * topN];
            var memStat = 1;
            try
            {
                IntPtr cValuesPtr = cValuesLoc.AddrOfPinnedObject();
                IntPtr cIdxPtr = cIdxLoc.AddrOfPinnedObject();
                IntPtr csrRowoffsetsPtr = csrRowoffsetsLoc.AddrOfPinnedObject();
                IntPtr csrIdxPtr = csrIdxLoc.AddrOfPinnedObject();
                IntPtr sValuesPtr = sValuesLoc.AddrOfPinnedObject();
                IntPtr sIdxPtr = sIdxLoc.AddrOfPinnedObject();

                IntPtr resultEigen = findTopCandidates(cValuesPtr, cIdxPtr, sValuesPtr, sIdxPtr,
                                                       candidateValues.Length, candidatesIdx.Length, spectraValues.Length, spectraIdx.Length,
                                                       topN, (float) 0.0, NORMALIZE, USE_GAUSSIAN, 0);

                Marshal.Copy(resultEigen, resultArrayEigen, 0, spectraIdx.Length * topN);

                memStat = releaseMemory(resultEigen);

                IntPtr resultCuda = findTopCandidatesCuda(csrRowoffsetsPtr, csrIdxPtr,
                                                          sValuesPtr, sIdxPtr,
                                                          csrRowoffsets.Length, csrIdx.Length,
                                                          spectraValues.Length, spectraIdx.Length,
                                                          topN, (float) 0.0);

                Marshal.Copy(resultCuda, resultArrayCuda, 0, spectraIdx.Length * topN);

                memStat = releaseMemoryCuda(resultCuda);
            }
            catch (Exception ex)
            {
                Console.WriteLine("Something went wrong:");
                Console.WriteLine(ex.ToString());
            }
            finally
            {
                if (cValuesLoc.IsAllocated) { cValuesLoc.Free(); }
                if (cIdxLoc.IsAllocated) { cIdxLoc.Free(); }
                if (csrRowoffsetsLoc.IsAllocated) { csrRowoffsetsLoc.Free(); }
                if (csrIdxLoc.IsAllocated) { csrIdxLoc.Free(); }
                if (sValuesLoc.IsAllocated) { sValuesLoc.Free(); }
                if (sIdxLoc.IsAllocated) { sIdxLoc.Free(); }
            }

            // end time c++ call
            sw.Stop();

            for (int i = 0; i < spectraIdx.Length * topN; i++)
            {
                Console.WriteLine($"Eigen: {resultArrayEigen[i]}");
            }
            for (int i = 0; i < spectraIdx.Length * topN; i++)
            {
                Console.WriteLine($"CUDA: {resultArrayCuda[i]}");
            }

            Console.WriteLine($"MemStat: {memStat}");
            Console.WriteLine("Time for candidate search (SpMV):");
            Console.WriteLine(sw.Elapsed.TotalSeconds.ToString());

            //
            GC.Collect();
            GC.WaitForPendingFinalizers();

            return 0;
        }
    }
}
