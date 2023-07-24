using System.Diagnostics;
using System.Runtime.InteropServices;

namespace FHOOE_IMP.MS_Annika.Utils.NonCleavableSearch
{
    public partial class DataLoader
    {
        const int MASS_RANGE = 1300;
        const int MASS_MULTIPLIER = 100;
        const int ENCODING_SIZE = MASS_RANGE * MASS_MULTIPLIER;
        const bool NORMALIZE = true;
        const bool USE_GAUSSIAN = true;
        const int BATCH_SIZE = 100;

        public static void Main(string[] args)
        {
            // parameter defaults
            var mode = "Eigen";
            var nrCandidates = 5000000;
            var nrSpectra = 199;
            var topN = 20;
            var seed = 1337;
            var r = new Random(seed);

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

            // call subroutine for specified mode
            if (mode == "Cuda")
            {
                var status = Cuda(nrCandidates, nrSpectra, topN, r, false);
                Console.WriteLine($"Cuda routine exited with status: {status}");
            }
            else if (mode == "CudaB")
            {
                var status = Cuda(nrCandidates, nrSpectra, topN, r, true);
                Console.WriteLine($"Cuda routine exited with status: {status}");
            }
            else if (mode == "Eigen")
            {
                var status = Eigen(nrCandidates, nrSpectra, topN, r, false);
                Console.WriteLine($"Eigen routine exited with status: {status}");
            }
            else if (mode == "EigenB")
            {
                var status = Eigen(nrCandidates, nrSpectra, topN, r, true);
                Console.WriteLine($"Eigen routine exited with status: {status}");
            }
            else if (mode == "Compare")
            {
                var status = Compare(nrCandidates, nrSpectra, topN, r);
                Console.WriteLine($"Compare routine exited with status: {status}");
            }
            else if (mode == "CompareD")
            {
                var status = DeterministicCompare();
                Console.WriteLine($"Deterministic compare routine exited with status: {status}");
            }
            else
            {
                Console.WriteLine("No mode selected, has to be one of: Eigen, EigenB, Cuda, Compare.");
            }
           
            Console.WriteLine("Done!");

            return;
        }
    }
}