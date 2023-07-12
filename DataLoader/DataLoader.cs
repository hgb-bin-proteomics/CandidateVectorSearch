using System.Diagnostics;
using System.Runtime.InteropServices;

namespace FHOOE_IMP.MS_Annika.Utils.NonCleavableSearch
{
    public partial class DataLoader
    {
        const int MASS_RANGE = 1300;
        const int MASS_MULTIPLIER = 100;
        const int ENCODING_SIZE = MASS_RANGE * MASS_MULTIPLIER;

        public static void Main(string[] args)
        {
            // parameter defaults
            var mode = "Eigen";
            var nrCandidates = 5000000;
            var nrSpectra = 100;
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
                var status = Cuda(nrCandidates, nrSpectra, topN, r);
                Console.WriteLine($"Cuda routine exited with status: {status}");
            }
            else if (mode == "Eigen")
            {
                var status = Eigen(nrCandidates, nrSpectra, topN, r);
                Console.WriteLine($"Eigen routine exited with status: {status}");
            }
            else
            {
                var status = Compare(nrCandidates, nrSpectra, topN, r);
                Console.WriteLine($"Compare routine exited with status: {status}");
            }
           
            Console.WriteLine("Done!");

            return;
        }
    }
}