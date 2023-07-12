using System.Diagnostics;
using System.Runtime.InteropServices;

namespace FHOOE_IMP.MS_Annika.Utils.NonCleavableSearch
{
    public partial class DataLoader
    {
        public static int DeterministicCompare()
        {
            // EXAMPLE PROBLEM
            // [ [0, 5, 0, 0, 1]        - 2
            //   [1, 3, 2, 1, 0]        - 4
            //   [0, 1, 8, 9, 0]        - 3
            //   [0, 4, 0, 0, 1]        - 2
            //   [1, 2, 3, 4, 5] ]      - 5
            //
            // [1, 2, 3, 4, 5]
            // [6, 7, 8, 9, 0]
            // 
            // Result 1:
            // [15, 17, 62, 13, 55] = [7.5, 4.25, 20.6, 6.5, 11] = [2, 4, 0, 3, 1]
            //
            // Result 2;
            // [35, 52, 152, 28, 80] = [17.5, 13, 50.6, 14, 16] = [2, 0, 4, 3, 1]
            // generate candidate vectors
            //                                0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
            var candidateValues = new int[] { 5, 1, 1, 3, 2, 1, 1, 8, 9, 4, 1, 1, 2, 3, 4, 5 };
            var candidatesIdx = new int[] { 0, 2, 6, 9, 11 };
            var csrRowoffsets = new int[] { 0, 2, 6, 9, 11, 16 };
            var csrIdx = new int[] { 1, 4, 0, 1, 2, 3, 1, 2, 3, 1, 4, 0, 1, 2, 3, 4 };
            var csrValues = new float[] { 5/2, 1/2, 1/4, 3/4, 2/4, 1/4, 1/3, 8/3, 9/3, 4/2, 1/2, 1/5, 2/5, 3/5, 4/5, 5/5 };

            // generate spectra vectors
            var spectraValues = new int[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
            var spectraIdx = new int[] { 0, 5 };

            // topN
            var topN = 5;

            // time c++ call
            var sw = Stopwatch.StartNew();

            // get pointer addresses and call c++ function
            var cValuesLoc = GCHandle.Alloc(candidateValues, GCHandleType.Pinned);
            var cIdxLoc = GCHandle.Alloc(candidatesIdx, GCHandleType.Pinned);
            var csrRowoffsetsLoc = GCHandle.Alloc(csrRowoffsets, GCHandleType.Pinned);
            var csrIdxLoc = GCHandle.Alloc(csrIdx, GCHandleType.Pinned);
            var csrValuesLoc = GCHandle.Alloc(csrValues, GCHandleType.Pinned);
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
                IntPtr csrValuesPtr = csrValuesLoc.AddrOfPinnedObject();
                IntPtr sValuesPtr = sValuesLoc.AddrOfPinnedObject();
                IntPtr sIdxPtr = sIdxLoc.AddrOfPinnedObject();

                IntPtr resultEigen = findTopCandidates(cValuesPtr, cIdxPtr, sValuesPtr, sIdxPtr,
                                                       candidateValues.Length, candidatesIdx.Length, spectraValues.Length, spectraIdx.Length,
                                                       topN, (float) 0.0);

                Marshal.Copy(resultEigen, resultArrayEigen, 0, spectraIdx.Length * topN);

                memStat = releaseMemory(resultEigen);

                IntPtr resultCuda = findTopCandidatesCuda(csrRowoffsetsPtr, csrIdxPtr, csrValuesPtr,
                                                          sValuesPtr, sIdxPtr,
                                                          csrRowoffsets.Length, csrValues.Length,
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
                if (csrValuesLoc.IsAllocated) { csrValuesLoc.Free(); }
                if (sValuesLoc.IsAllocated) { sValuesLoc.Free(); }
                if (sIdxLoc.IsAllocated) { sIdxLoc.Free(); }
            }

            // end time c++ call
            sw.Stop();

            for (int i = 0; i < spectraIdx.Length * topN; i++)
            {
                Console.WriteLine(resultArrayEigen[i]);
                Console.WriteLine(resultArrayCuda[i]);
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
