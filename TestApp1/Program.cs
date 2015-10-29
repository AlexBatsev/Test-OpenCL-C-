using System;
using System.IO;
using System.Linq;
using Cloo;

namespace TestApp1
{
    internal static class Program
    {
        private static void Main(string[] args)
        {
            var gpu = new GpuProgram("fft256.cl", "fft256");

            var xy =
                File.ReadAllText(@"..\..\out.txt")
                    .Split(new[] {"\n"}, StringSplitOptions.None)
                    .Where(s => s.Length > 1)
                    .Select(s => s.Split(new[] {"\t"}, StringSplitOptions.None).Select(float.Parse).ToArray())
                    .ToArray();

            var x = xy.SelectMany(floats => new[] {floats[0], 0.0f}).ToArray();
            var y = xy.SelectMany(floats => new[] {floats[1], floats[2]}).ToArray();

            var inBuffer = new ComputeBuffer<float>(gpu.Context,
                ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.UseHostPointer, x);
            var outBuffer = new ComputeBuffer<float>(gpu.Context, ComputeMemoryFlags.ReadWrite, y.Length);
            gpu.Queue.WriteToBuffer(x, inBuffer, true, null);
            gpu.Kernel.SetMemoryArgument(0, inBuffer);
            gpu.Kernel.SetMemoryArgument(1, outBuffer);
            gpu.Queue.Execute(gpu.Kernel, null, new long[] {64}, new long[] {64}, null);
            gpu.Queue.Finish();
            var y1 = new float[y.Length];
            gpu.Queue.ReadFromBuffer(outBuffer, ref y1, true, null);

        }

        private static void TestNativeCos()
        {
            var gpu = new GpuProgram("kernel1.cl", "helloWorld");

            const int n = 100;
            var fullCos = new float[n];
            var nativeCos = new float[n];
            var doubleCos = Enumerable.Range(0, n).Select(i => Math.Cos(Math.PI * 2.0 / n * i)).ToArray();
            var fullCosBuffer = new ComputeBuffer<float>(gpu.Context,
                ComputeMemoryFlags.WriteOnly | ComputeMemoryFlags.UseHostPointer, fullCos);
            var nativeCosBuffer = new ComputeBuffer<float>(gpu.Context,
                ComputeMemoryFlags.WriteOnly | ComputeMemoryFlags.UseHostPointer, nativeCos);

            gpu.Kernel.SetMemoryArgument(0, fullCosBuffer); // set the integer array
            gpu.Kernel.SetMemoryArgument(1, nativeCosBuffer); // set the integer array

            gpu.Queue.Execute(gpu.Kernel, null, new long[] {n}, new long[] {n}, null);
            gpu.Queue.ReadFromBuffer(fullCosBuffer, ref fullCos, true, null);
            gpu.Queue.ReadFromBuffer(nativeCosBuffer, ref nativeCos, true, null);
            // wait for completion

            var diffs = fullCos.Zip(nativeCos, (f, f1) => f1 != 0.0 ? Math.Abs((f - f1) / f1) : 0.0f).ToArray();
            var maxDiff = diffs.Max();
            gpu.Queue.Finish();
        }
    }
}