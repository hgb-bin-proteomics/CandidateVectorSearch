using System.Diagnostics;
using System.Runtime.InteropServices;

namespace CandidateVectorSearch
{
    /// <summary>
    /// Prototype implementation to test C++/C# marshalling and the different matrix multiplication approaches.
    /// </summary>
    public partial class DataLoader
    {
        /// <summary>
        /// Fixed parameter determined by the CandidateVectorSearch DLLs.
        /// </summary>
        const int MASS_RANGE = 5000;

        /// <summary>
        /// Fixed parameter determined by the CandidateVectorSearch DLLs.
        /// </summary>
        const int MASS_MULTIPLIER = 100;

        /// <summary>
        /// Fixed parameter determined by the CandidateVectorSearch DLLs.
        /// </summary>
        const int ENCODING_SIZE = MASS_RANGE * MASS_MULTIPLIER;
        
        /// <summary>
        /// Whether or not to normalize scores (true to simulate worst-case scenario).
        /// </summary>
        const bool NORMALIZE = true;

        /// <summary>
        /// Whether or not to model peaks as gaussian distributions (true to simulate worst-case scenario).
        /// </summary>
        const bool USE_GAUSSIAN = true;

        /// <summary>
        /// Main function to be executed when calling the executable.\n
        /// The routine to be called depends on the commandline arguments passed to the function.\n
        /// The commandline call has to look like this:\n
        /// DataLoader.exe [(string)Mode][(int)NrCandidates][(int)NrSpectra][(int)TopN][(int)BatchSize]
        /// </summary>
        /// <param name="args">Array of commandline arguments.</param>
        public static void Main(string[] args)
        {
            // parameter defaults
            var mode = "Eigen";
            var nrCandidates = 5000000;
            var nrSpectra = 199;
            var topN = 20;
            var batchSize = 100;
            //var seed = 1337;
            var r = new Random(); //new Random(seed);

            // parse arguments
            if (args.Length > 0)
            {
                mode = args[0];
            }
            if (args.Length > 1)
            {
                nrCandidates = int.Parse(args[1]);
            }
            if (args.Length > 2)
            {
                nrSpectra = int.Parse(args[2]);
            }
            if (args.Length > 3)
            {
                topN = int.Parse(args[3]);
            }
            if (args.Length > 4)
            {
                batchSize = int.Parse(args[4]);
            }

            // call subroutine for specified mode
            if (mode == "Cuda")
            {
                var status = Cuda(nrCandidates, nrSpectra, topN, batchSize, r, false, 1);
                Console.WriteLine($"Cuda routine exited with status: {status}");
            }
            else if (mode == "CudaB")
            {
                var status = Cuda(nrCandidates, nrSpectra, topN, batchSize, r, true, 1);
                Console.WriteLine($"Cuda routine exited with status: {status}");
            }
            else if (mode == "CudaBAlt")
            {
                var status = Cuda(nrCandidates, nrSpectra, topN, batchSize, r, true, 2);
                Console.WriteLine($"Cuda routine exited with status: {status}");
            }
            else if (mode == "Eigen")
            {
                var status = Eigen(nrCandidates, nrSpectra, topN, batchSize, r, false, false);
                Console.WriteLine($"Eigen routine exited with status: {status}");
            }
            else if (mode == "EigenInt")
            {
                var status = Eigen(nrCandidates, nrSpectra, topN, batchSize, r, false, false, true);
                Console.WriteLine($"Eigen routine exited with status: {status}");
            }
            else if (mode == "EigenB")
            {
                var status = Eigen(nrCandidates, nrSpectra, topN, batchSize, r, true, false);
                Console.WriteLine($"Eigen routine exited with status: {status}");
            }
            else if (mode == "EigenIntB")
            {
                var status = Eigen(nrCandidates, nrSpectra, topN, batchSize, r, true, false, true);
                Console.WriteLine($"Eigen routine exited with status: {status}");
            }
            else if (mode == "EigenS")
            {
                var status = Eigen(nrCandidates, nrSpectra, topN, batchSize, r, false, true);
                Console.WriteLine($"Eigen routine exited with status: {status}");
            }
            else if (mode == "EigenSInt")
            {
                var status = Eigen(nrCandidates, nrSpectra, topN, batchSize, r, false, true, true);
                Console.WriteLine($"Eigen routine exited with status: {status}");
            }
            else if (mode == "EigenSB")
            {
                var status = Eigen(nrCandidates, nrSpectra, topN, batchSize, r, true, true);
                Console.WriteLine($"Eigen routine exited with status: {status}");
            }
            else if (mode == "EigenSIntB")
            {
                var status = Eigen(nrCandidates, nrSpectra, topN, batchSize, r, true, true, true);
                Console.WriteLine($"Eigen routine exited with status: {status}");
            }
            else if (mode == "Benchmark")
            {
                var status = Benchmark(nrCandidates, nrSpectra, topN, batchSize, r);
                Console.WriteLine($"Compare routine exited with status: {status}");
            }
            else if (mode == "Compare")
            {
                var status = Compare(nrCandidates, nrSpectra, topN, batchSize, r);
                Console.WriteLine($"Compare routine exited with status: {status}");
            }
            else if (mode == "CompareD")
            {
                var status = DeterministicCompare();
                Console.WriteLine($"Deterministic compare routine exited with status: {status}");
            }
            else
            {
                Console.WriteLine("No mode selected, has to be one of: Eigen(S)(Int), Eigen(S)(Int)B, Cuda(B/BAlt), Compare, Benchmark.");
            }
           
            Console.WriteLine("Done!");

            return;
        }
    }
}