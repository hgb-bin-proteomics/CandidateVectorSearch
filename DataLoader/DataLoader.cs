using System.Diagnostics;
using System.Runtime.InteropServices;

namespace FHOOE_IMP.MS_Annika.Utils.NonCleavableSearch
{
    public class DataLoader
    {
        const int MASS_RANGE = 1300;
        const int MASS_MULTIPLIER = 100;
        const int ENCODING_SIZE = MASS_RANGE * MASS_MULTIPLIER;

        const string dll = "VectorSearch.dll";
        [DllImport(dll, CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr findTopCandidates(IntPtr cV, IntPtr cI, IntPtr sV, IntPtr sI,
                                                       int cVL, int cIL, int sVL, int sIL,
                                                       int n, float tolerance);

        [DllImport(dll, CallingConvention = CallingConvention.Cdecl)]
        private static extern int releaseMemory(IntPtr result);

        public static void Main(string[] args)
        {
            var nrCandidates = 5000000;
            var nrSpectra = 100;
            var topN = 20;
            var seed = 1337;
            var r = new Random(seed);

            //
            if (args.Length > 0)
            {
                nrCandidates = int.Parse(args[0]);
            }

            if (args.Length > 1)
            {
                nrSpectra = int.Parse(args[1]);
            }

            if (args.Length > 2)
            {
                topN = int.Parse(args[2]);
            }

            //
            var candidateValues = new int[nrCandidates * 100];
            var candidatesIdx = new int[nrCandidates];
            var currentIdx = 0;
            for (int i = 0; i < candidateValues.Length; i+=100)
            {
                candidatesIdx[currentIdx] = i;
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
                    candidateValues[i+j] = tmpValues[j];
                }
                currentIdx++;
                if (currentIdx % 5000 == 0)
                {
                    Console.WriteLine($"Generated {currentIdx} candidates...");
                }
            }

            //
            var spectraValues = new int[nrSpectra * 500];
            var spectraIdx = new int[nrSpectra];
            currentIdx = 0;
            for (int i = 0; i < spectraValues.Length; i+=500)
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

            //
            var sw = Stopwatch.StartNew();

            //
            var cValuesLoc = GCHandle.Alloc(candidateValues, GCHandleType.Pinned);
            var cIdxLoc = GCHandle.Alloc(candidatesIdx, GCHandleType.Pinned);
            var sValuesLoc = GCHandle.Alloc(spectraValues, GCHandleType.Pinned);
            var sIdxLoc = GCHandle.Alloc(spectraIdx, GCHandleType.Pinned);
            var resultArray = new int[spectraIdx.Length * topN];
            var memStat = 1;
            try
            {
                IntPtr cValuesPtr = cValuesLoc.AddrOfPinnedObject();
                IntPtr cIdxPtr = cIdxLoc.AddrOfPinnedObject();
                IntPtr sValuesPtr = sValuesLoc.AddrOfPinnedObject();
                IntPtr sIdxPtr = sIdxLoc.AddrOfPinnedObject();

                IntPtr result = findTopCandidates(cValuesPtr, cIdxPtr, sValuesPtr, sIdxPtr,
                                                  candidateValues.Length, candidatesIdx.Length, spectraValues.Length, spectraIdx.Length,
                                                  topN, (float) 0.02);

                Marshal.Copy(result, resultArray, 0, spectraIdx.Length * topN);

                memStat = releaseMemory(result);
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
                if (sValuesLoc.IsAllocated) { sValuesLoc.Free(); }
                if (sIdxLoc.IsAllocated) { sIdxLoc.Free(); }
            }

            //
            sw.Stop();

            for (int i = 0; i < topN; i++)
            {
                Console.WriteLine(resultArray[i]);
            }

            Console.WriteLine($"MemStat: {memStat}");
            Console.WriteLine("Time for candidate search (SpMV):");
            Console.WriteLine(sw.Elapsed.TotalSeconds.ToString());

            //
            GC.Collect();
            GC.WaitForPendingFinalizers();

            Console.WriteLine("Done!");

            return;
        }
    }
}