using System.Diagnostics;
using System.Runtime.InteropServices;

namespace CandidateVectorSearch
{
    public partial class DataLoader
    {
        public static int Benchmark(int nrCandidates, int nrSpectra, int topN, int batchSize, Random r)
        {
            // generate candidate vectors
            var candidateValues = new int[nrCandidates * 100];
            var candidatesIdx = new int[nrCandidates];
            var csrRowoffsets = new int[nrCandidates + 1];
            var csrIdx = new int[nrCandidates * 100];
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

            // get pointer addresses and call c++ function
            var cValuesLoc = GCHandle.Alloc(candidateValues, GCHandleType.Pinned);
            var cIdxLoc = GCHandle.Alloc(candidatesIdx, GCHandleType.Pinned);
            var csrRowoffsetsLoc = GCHandle.Alloc(csrRowoffsets, GCHandleType.Pinned);
            var csrIdxLoc = GCHandle.Alloc(csrIdx, GCHandleType.Pinned);
            var sValuesLoc = GCHandle.Alloc(spectraValues, GCHandleType.Pinned);
            var sIdxLoc = GCHandle.Alloc(spectraIdx, GCHandleType.Pinned);
            var resultArrayEigen = new int[spectraIdx.Length * topN];
            var resultArrayEigenInt = new int[spectraIdx.Length * topN];
            var resultArrayEigen2 = new int[spectraIdx.Length * topN];
            var resultArrayEigen2Int = new int[spectraIdx.Length * topN];
            var resultArrayEigenB = new int[spectraIdx.Length * topN];
            var resultArrayEigenBInt = new int[spectraIdx.Length * topN];
            var resultArrayEigenB2 = new int[spectraIdx.Length * topN];
            var resultArrayEigenB2Int = new int[spectraIdx.Length * topN];
            var resultArrayCuda = new int[spectraIdx.Length * topN];
            var resultArrayCudaB = new int[spectraIdx.Length * topN];
            var resultArrayCudaB2 = new int[spectraIdx.Length * topN];
            var memStat = 1;
            try
            {
                IntPtr cValuesPtr = cValuesLoc.AddrOfPinnedObject();
                IntPtr cIdxPtr = cIdxLoc.AddrOfPinnedObject();
                IntPtr csrRowoffsetsPtr = csrRowoffsetsLoc.AddrOfPinnedObject();
                IntPtr csrIdxPtr = csrIdxLoc.AddrOfPinnedObject();
                IntPtr sValuesPtr = sValuesLoc.AddrOfPinnedObject();
                IntPtr sIdxPtr = sIdxLoc.AddrOfPinnedObject();

                var sw1 = Stopwatch.StartNew();

                IntPtr resultEigen = findTopCandidates(cValuesPtr, cIdxPtr, sValuesPtr, sIdxPtr,
                                                       candidateValues.Length, candidatesIdx.Length, spectraValues.Length, spectraIdx.Length,
                                                       topN, (float) 0.02, NORMALIZE, USE_GAUSSIAN, 0, 0);

                Marshal.Copy(resultEigen, resultArrayEigen, 0, spectraIdx.Length * topN);

                memStat = releaseMemory(resultEigen);

                sw1.Stop();

                Console.WriteLine("Time for candidate search Eigen SpM*SpV:");
                Console.WriteLine(sw1.Elapsed.TotalSeconds.ToString());

                var sw1Int = Stopwatch.StartNew();

                IntPtr resultEigenInt = findTopCandidatesInt(cValuesPtr, cIdxPtr, sValuesPtr, sIdxPtr,
                                                             candidateValues.Length, candidatesIdx.Length, spectraValues.Length, spectraIdx.Length,
                                                             topN, (float) 0.02, NORMALIZE, USE_GAUSSIAN, 0, 0);

                Marshal.Copy(resultEigenInt, resultArrayEigenInt, 0, spectraIdx.Length * topN);

                memStat = releaseMemory(resultEigenInt);

                sw1Int.Stop();

                Console.WriteLine("Time for candidate search Eigen SpM*SpV (int):");
                Console.WriteLine(sw1Int.Elapsed.TotalSeconds.ToString());

                var sw2 = Stopwatch.StartNew();

                IntPtr resultEigen2 = findTopCandidates2(cValuesPtr, cIdxPtr, sValuesPtr, sIdxPtr,
                                                         candidateValues.Length, candidatesIdx.Length, spectraValues.Length, spectraIdx.Length,
                                                         topN, (float) 0.02, NORMALIZE, USE_GAUSSIAN, 0, 0);

                Marshal.Copy(resultEigen2, resultArrayEigen2, 0, spectraIdx.Length * topN);

                memStat = releaseMemory(resultEigen2);

                sw2.Stop();

                Console.WriteLine("Time for candidate search Eigen SpM*V:");
                Console.WriteLine(sw2.Elapsed.TotalSeconds.ToString());

                var sw2Int = Stopwatch.StartNew();

                IntPtr resultEigen2Int = findTopCandidates2Int(cValuesPtr, cIdxPtr, sValuesPtr, sIdxPtr,
                                                               candidateValues.Length, candidatesIdx.Length, spectraValues.Length, spectraIdx.Length,
                                                               topN, (float) 0.02, NORMALIZE, USE_GAUSSIAN, 0, 0);

                Marshal.Copy(resultEigen2Int, resultArrayEigen2Int, 0, spectraIdx.Length * topN);

                memStat = releaseMemory(resultEigen2Int);

                sw2Int.Stop();

                Console.WriteLine("Time for candidate search Eigen SpM*V (int):");
                Console.WriteLine(sw2Int.Elapsed.TotalSeconds.ToString());

                var sw3 = Stopwatch.StartNew();

                IntPtr resultEigenB = findTopCandidatesBatched(cValuesPtr, cIdxPtr, sValuesPtr, sIdxPtr,
                                                               candidateValues.Length, candidatesIdx.Length, spectraValues.Length, spectraIdx.Length,
                                                               topN, (float) 0.02, NORMALIZE, USE_GAUSSIAN, batchSize, 0, 0);

                Marshal.Copy(resultEigenB, resultArrayEigenB, 0, spectraIdx.Length * topN);

                memStat = releaseMemory(resultEigenB);

                sw3.Stop();

                Console.WriteLine("Time for candidate search Eigen SpM*SpM:");
                Console.WriteLine(sw3.Elapsed.TotalSeconds.ToString());

                var sw3Int = Stopwatch.StartNew();

                IntPtr resultEigenBInt = findTopCandidatesBatchedInt(cValuesPtr, cIdxPtr, sValuesPtr, sIdxPtr,
                                                                     candidateValues.Length, candidatesIdx.Length, spectraValues.Length, spectraIdx.Length,
                                                                     topN, (float) 0.02, NORMALIZE, USE_GAUSSIAN, batchSize, 0, 0);

                Marshal.Copy(resultEigenBInt, resultArrayEigenBInt, 0, spectraIdx.Length * topN);

                memStat = releaseMemory(resultEigenBInt);

                sw3Int.Stop();

                Console.WriteLine("Time for candidate search Eigen SpM*SpM (int):");
                Console.WriteLine(sw3Int.Elapsed.TotalSeconds.ToString());

                var sw4 = Stopwatch.StartNew();

                IntPtr resultEigenB2 = findTopCandidatesBatched2(cValuesPtr, cIdxPtr, sValuesPtr, sIdxPtr,
                                                                 candidateValues.Length, candidatesIdx.Length, spectraValues.Length, spectraIdx.Length,
                                                                 topN, (float) 0.02, NORMALIZE, USE_GAUSSIAN, batchSize, 0, 0);

                Marshal.Copy(resultEigenB2, resultArrayEigenB2, 0, spectraIdx.Length * topN);

                memStat = releaseMemory(resultEigenB2);

                sw4.Stop();

                Console.WriteLine("Time for candidate search Eigen SpM*M:");
                Console.WriteLine(sw4.Elapsed.TotalSeconds.ToString());

                var sw4Int = Stopwatch.StartNew();

                IntPtr resultEigenB2Int = findTopCandidatesBatched2Int(cValuesPtr, cIdxPtr, sValuesPtr, sIdxPtr,
                                                                       candidateValues.Length, candidatesIdx.Length, spectraValues.Length, spectraIdx.Length,
                                                                       topN, (float) 0.02, NORMALIZE, USE_GAUSSIAN, batchSize, 0, 0);

                Marshal.Copy(resultEigenB2Int, resultArrayEigenB2Int, 0, spectraIdx.Length * topN);

                memStat = releaseMemory(resultEigenB2Int);

                sw4Int.Stop();

                Console.WriteLine("Time for candidate search Eigen SpM*M (int):");
                Console.WriteLine(sw4Int.Elapsed.TotalSeconds.ToString());

                var sw5 = Stopwatch.StartNew();

                IntPtr resultCuda = findTopCandidatesCuda(csrRowoffsetsPtr, csrIdxPtr,
                                                          sValuesPtr, sIdxPtr,
                                                          csrRowoffsets.Length, csrIdx.Length,
                                                          spectraValues.Length, spectraIdx.Length,
                                                          topN, (float) 0.02, NORMALIZE, USE_GAUSSIAN, 0);

                Marshal.Copy(resultCuda, resultArrayCuda, 0, spectraIdx.Length * topN);

                memStat = releaseMemoryCuda(resultCuda);

                sw5.Stop();

                Console.WriteLine("Time for candidate search Cuda SpMV:");
                Console.WriteLine(sw5.Elapsed.TotalSeconds.ToString());

                var sw6 = Stopwatch.StartNew();

                IntPtr resultCudaB2 = findTopCandidatesCudaBatched2(csrRowoffsetsPtr, csrIdxPtr,
                                                                    sValuesPtr, sIdxPtr,
                                                                    csrRowoffsets.Length, csrIdx.Length,
                                                                    spectraValues.Length, spectraIdx.Length,
                                                                    topN, (float) 0.02, NORMALIZE, USE_GAUSSIAN, batchSize, 0);

                Marshal.Copy(resultCudaB2, resultArrayCudaB2, 0, spectraIdx.Length * topN);

                memStat = releaseMemoryCuda(resultCudaB2);

                sw6.Stop();

                Console.WriteLine("Time for candidate search Cuda SpMM:");
                Console.WriteLine(sw6.Elapsed.TotalSeconds.ToString());

                var sw7 = Stopwatch.StartNew();

                IntPtr resultCudaB = findTopCandidatesCudaBatched(csrRowoffsetsPtr, csrIdxPtr,
                                                                  sValuesPtr, sIdxPtr,
                                                                  csrRowoffsets.Length, csrIdx.Length,
                                                                  spectraValues.Length, spectraIdx.Length,
                                                                  topN, (float) 0.02, NORMALIZE, USE_GAUSSIAN, batchSize, 0);

                Marshal.Copy(resultCudaB, resultArrayCudaB, 0, spectraIdx.Length * topN);

                memStat = releaseMemoryCuda(resultCudaB);

                sw7.Stop();
                
                Console.WriteLine("Time for candidate search Cuda SpGEMM:");
                Console.WriteLine(sw7.Elapsed.TotalSeconds.ToString());
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

            Console.WriteLine($"Top 5 of the first spectrum:");

            for (int i = 0; i < 5; i++)
            {
                Console.WriteLine($"eSpVf32: {resultArrayEigen[i]}");
                Console.WriteLine($"eSpVi32: {resultArrayEigenInt[i]}");
                Console.WriteLine($"eDnVf32: {resultArrayEigen2[i]}");
                Console.WriteLine($"eDnVi32: {resultArrayEigen2Int[i]}");
                Console.WriteLine($"eSpMf32: {resultArrayEigenB[i]}");
                Console.WriteLine($"eSpMi32: {resultArrayEigenBInt[i]}");
                Console.WriteLine($"eDnMf32: {resultArrayEigenB2[i]}");
                Console.WriteLine($"eDnMi32: {resultArrayEigenB2Int[i]}");
                Console.WriteLine($"cDnVf32: {resultArrayCuda[i]}");
                Console.WriteLine($"cSpMf32: {resultArrayCudaB[i]}");
                Console.WriteLine($"cDnMf32: {resultArrayCudaB2[i]}");
                Console.WriteLine("-----");
            }

            Console.WriteLine($"Top 5 of the last spectrum:");

            for (int i = spectraIdx.Length * topN - topN; i < spectraIdx.Length * topN - topN + 5; i++)
            {
                Console.WriteLine($"eSpVf32: {resultArrayEigen[i]}");
                Console.WriteLine($"eSpVi32: {resultArrayEigenInt[i]}");
                Console.WriteLine($"eDnVf32: {resultArrayEigen2[i]}");
                Console.WriteLine($"eDnVi32: {resultArrayEigen2Int[i]}");
                Console.WriteLine($"eSpMf32: {resultArrayEigenB[i]}");
                Console.WriteLine($"eSpMi32: {resultArrayEigenBInt[i]}");
                Console.WriteLine($"eDnMf32: {resultArrayEigenB2[i]}");
                Console.WriteLine($"eDnMi32: {resultArrayEigenB2Int[i]}");
                Console.WriteLine($"cDnVf32: {resultArrayCuda[i]}");
                Console.WriteLine($"cSpMf32: {resultArrayCudaB[i]}");
                Console.WriteLine($"cDnMf32: {resultArrayCudaB2[i]}");
                Console.WriteLine("-----");
            }

            Console.WriteLine($"MemStat: {memStat}");

            //
            GC.Collect();
            GC.WaitForPendingFinalizers();

            return 0;
        }
    }
}
