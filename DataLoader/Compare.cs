using System.Diagnostics;
using System.Runtime.InteropServices;

namespace FHOOE_IMP.MS_Annika.Utils.NonCleavableSearch
{
    public partial class DataLoader
    {
        public static int Compare(int nrCandidates, int nrSpectra, int topN, Random r)
        {
            // generate candidate vectors
            var candidateValues = new int[nrCandidates * 100];
            var candidatesIdx = new int[nrCandidates];
            var csrRowoffsets = new int[nrCandidates + 1];
            var csrIdx = new int[nrCandidates * 100];
            var csrValues = new float[nrCandidates * 100];
            var currentIdx = 0;
            for (int i = 0; i < candidateValues.Length; i += 100)
            {
                candidatesIdx[currentIdx] = i;
                csrRowoffsets[currentIdx] = i;
                var tmpValues = new int[100];
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
                    candidateValues[i + j] = tmpValues[j];
                    csrIdx[i + j] = tmpValues[j];
                    csrValues[i + j] = (float)(1.0 / 100.0); // this needs to be changed for the actual NNZ per row
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
                                                       topN, (float) 0.02);

                Marshal.Copy(resultEigen, resultArrayEigen, 0, spectraIdx.Length * topN);

                memStat = releaseMemory(resultEigen);

                IntPtr resultCuda = findTopCandidatesCuda(csrRowoffsetsPtr, csrIdxPtr, csrValuesPtr,
                                                          sValuesPtr, sIdxPtr,
                                                          csrRowoffsets.Length, csrValues.Length,
                                                          spectraValues.Length, spectraIdx.Length,
                                                          topN, (float) 0.02);

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

            for (int i = 0; i < topN; i++)
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
