using System;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Text;
using System.Threading;

namespace Bots
{
    // Samples NeuralEvaluate's inputs and outputs to JSONL, to compare the in-game
    // feature distribution with the training/validation one (see
    // experiments/compare_ingame_vs_val.py, experiments/verify_csharp_inference.py).
    // Enabled by SOT_DUMP_DIR; no overhead when unset.
    //
    // Concurrent games in one GameRunner process share its pid and so these per-pid
    // files: counters are Interlocked, appends are locked, and the caller's game_id
    // tells rows from different games apart.
    public static class FeatureDumper
    {
        private static readonly string? DumpDir =
            Environment.GetEnvironmentVariable("SOT_DUMP_DIR");

        public static readonly bool Enabled = !string.IsNullOrEmpty(DumpDir);

        private static readonly string MeansDumpPath = Enabled
            ? Path.Combine(DumpDir!, $"evals_{Process.GetCurrentProcess().Id}.jsonl")
            : string.Empty;

        private static readonly string FullDumpPath = Enabled
            ? Path.Combine(DumpDir!, $"full_{Process.GetCurrentProcess().Id}.jsonl")
            : string.Empty;

        private static readonly object _meansLock = new object();
        private static readonly object _fullLock = new object();

        private static long _evalCounter = 0;
        private static long _terminalEvalCounter = 0;
        private static long _fullEvalCounter = 0;

        static FeatureDumper()
        {
            if (Enabled)
            {
                Directory.CreateDirectory(DumpDir!);
            }
        }

        // Called from NeuralEvaluate after wp is computed. Returns true iff a row was
        // written to the evals file, so the caller can count rows dumped per game.
        //
        // Two independent samplers:
        //  - column-mean row -> evals_{pid}.jsonl: ~1 in 2000 evals, plus ~1 in 500
        //    terminal evals, which are rarer.
        //  - full node-matrix row -> full_{pid}.jsonl: ~1 in 50000 evals (~30 per game).
        //    Only this one can reproduce the forward pass, because the node encoder
        //    applies ReLU per card before pooling: mean(phi(x)) != phi(mean(x)).
        public static bool MaybeDump(int gameId, int turn, bool isTerminal, float[,] nodeFeatures, float[] globalFeatures, float csharpProb)
        {
            if (!Enabled) return false;

            bool dumpedMeans = MaybeDumpMeans(gameId, turn, isTerminal, nodeFeatures, globalFeatures, csharpProb);
            MaybeDumpFull(gameId, turn, isTerminal, nodeFeatures, globalFeatures, csharpProb);
            return dumpedMeans;
        }

        private static bool MaybeDumpMeans(int gameId, int turn, bool isTerminal, float[,] nodeFeatures, float[] globalFeatures, float csharpProb)
        {
            bool shouldDump = false;

            long n = Interlocked.Increment(ref _evalCounter);
            if (n % 2000 == 0) shouldDump = true;

            if (isTerminal)
            {
                long t = Interlocked.Increment(ref _terminalEvalCounter);
                if (t % 500 == 0) shouldDump = true;
            }

            if (!shouldDump) return false;

            float[] nodeMeans = ColumnMeans(nodeFeatures);
            int numNodes = nodeFeatures.GetLength(0);

            var sb = new StringBuilder();
            sb.Append("{\"game_id\":").Append(gameId.ToString(CultureInfo.InvariantCulture));
            sb.Append(",\"turn\":").Append(turn.ToString(CultureInfo.InvariantCulture));
            sb.Append(",\"is_terminal\":").Append(isTerminal ? "true" : "false");
            sb.Append(",\"num_nodes\":").Append(numNodes.ToString(CultureInfo.InvariantCulture));
            sb.Append(",\"csharp_prob\":").Append(csharpProb.ToString(CultureInfo.InvariantCulture));
            sb.Append(",\"global\":");
            AppendFloatArray(sb, globalFeatures, null);
            sb.Append(",\"node_means\":");
            AppendFloatArray(sb, nodeMeans, null);
            sb.Append('}');

            lock (_meansLock)
            {
                File.AppendAllText(MeansDumpPath, sb.ToString() + "\n");
            }
            return true;
        }

        private static void MaybeDumpFull(int gameId, int turn, bool isTerminal, float[,] nodeFeatures, float[] globalFeatures, float csharpProb)
        {
            long f = Interlocked.Increment(ref _fullEvalCounter);
            if (f % 50000 != 0) return;

            int numNodes = nodeFeatures.GetLength(0);
            int numFeatures = nodeFeatures.GetLength(1);

            // Round-trip precision (G9 for float32), so verify_csharp_inference.py can
            // replay the exact forward pass.
            const string fmt = "G9";

            var sb = new StringBuilder();
            sb.Append("{\"game_id\":").Append(gameId.ToString(CultureInfo.InvariantCulture));
            sb.Append(",\"turn\":").Append(turn.ToString(CultureInfo.InvariantCulture));
            sb.Append(",\"is_terminal\":").Append(isTerminal ? "true" : "false");
            sb.Append(",\"num_nodes\":").Append(numNodes.ToString(CultureInfo.InvariantCulture));
            sb.Append(",\"csharp_prob\":").Append(csharpProb.ToString(fmt, CultureInfo.InvariantCulture));
            sb.Append(",\"global\":");
            AppendFloatArray(sb, globalFeatures, fmt);
            sb.Append(",\"nodes\":[");
            for (int i = 0; i < numNodes; i++)
            {
                if (i > 0) sb.Append(',');
                sb.Append('[');
                for (int j = 0; j < numFeatures; j++)
                {
                    if (j > 0) sb.Append(',');
                    sb.Append(nodeFeatures[i, j].ToString(fmt, CultureInfo.InvariantCulture));
                }
                sb.Append(']');
            }
            sb.Append(']');
            sb.Append('}');

            lock (_fullLock)
            {
                File.AppendAllText(FullDumpPath, sb.ToString() + "\n");
            }
        }

        private static float[] ColumnMeans(float[,] nodeFeatures)
        {
            int numNodes = nodeFeatures.GetLength(0);
            int numFeatures = nodeFeatures.GetLength(1);
            float[] means = new float[numFeatures];
            if (numNodes == 0) return means;

            for (int j = 0; j < numFeatures; j++)
            {
                double sum = 0;
                for (int i = 0; i < numNodes; i++)
                    sum += nodeFeatures[i, j];
                means[j] = (float)(sum / numNodes);
            }
            return means;
        }

        private static void AppendFloatArray(StringBuilder sb, float[] values, string? format)
        {
            sb.Append('[');
            for (int i = 0; i < values.Length; i++)
            {
                if (i > 0) sb.Append(',');
                sb.Append(format == null
                    ? values[i].ToString(CultureInfo.InvariantCulture)
                    : values[i].ToString(format, CultureInfo.InvariantCulture));
            }
            sb.Append(']');
        }
    }
}
