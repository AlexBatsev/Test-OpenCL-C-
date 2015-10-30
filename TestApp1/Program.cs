using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Cloo;

namespace TestApp1
{
    internal static class Program
    {
        private const int n = 256;
        private static readonly int workSize = 64;
        private static readonly long[] localWorkSize = {workSize, 1};

        private static void Main(string[] args)
        {
            var gpu = new GpuProgram("fft256.cl", "fft256");

            const int nRows = 256;
            var x = new MyRand(-100.0, 100.0).GetNext(nRows * n, 2).Select(d => new Float2(d[0], d[1])).To2DArray(nRows);
            var yExact = Fft1DExact(x);

            var inBuffer = new ComputeBuffer<Float2>(gpu.Context, ComputeMemoryFlags.ReadOnly, n);
            var outBuffer = new ComputeBuffer<Float2>(gpu.Context, ComputeMemoryFlags.ReadWrite, yExact.Length);
            gpu.Queue.WriteToBuffer(x, inBuffer, true, new SysIntX2(0, 0), new SysIntX2(0, 0), new SysIntX2(nRows, n), null);
            gpu.Kernel.SetMemoryArgument(0, inBuffer);
            gpu.Kernel.SetMemoryArgument(1, outBuffer);
            for (int i = 0; i < 2; i++)
            {
                gpu.Queue.Execute(gpu.Kernel, null, new[] { workSize, (long)nRows }, localWorkSize, null);
                gpu.Queue.Finish();
            }
            //gpu.Queue.Execute(gpu.Kernel, null, new[] {workSize, (long) nRows}, localWorkSize, null);
            var y = new Float2[yExact.Length];
            gpu.Queue.ReadFromBuffer(outBuffer, ref y, true, null);
            var discrep =
                yExact.Cast<Float2>()
                    .Zip(y, (f1, f2) =>
                            new
                            {
                                exact = f1,
                                calculated = f2,
                                discrep = f2.X != 0.0f ? ((f1 - f2) / f2).Module : 0.0
                            })
                    .ToArray();
            var maxDiscrep = discrep.OrderBy(arg => arg.discrep).Last();
        }

        private static void Fft1D()
        {
            var gpu = new GpuProgram("fft256.cl", "fft256");

            var xy =
                File.ReadAllText(@"..\..\1D.txt")
                    .Split(new[] {"\n"}, StringSplitOptions.None)
                    .Where(s => s.Length > 1)
                    .Select(s => s.Split(new[] {"\t"}, StringSplitOptions.None).Select(float.Parse).ToArray())
                    .ToArray();

            var x = xy.Select(floats => new Float2(floats[0], 0.0f)).ToArray();
            var y = xy.Select(floats => new Float2(floats[1], floats[2])).ToArray();

            var inBuffer = new ComputeBuffer<Float2>(gpu.Context,
                ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.UseHostPointer, x);
            var outBuffer = new ComputeBuffer<Float2>(gpu.Context, ComputeMemoryFlags.ReadWrite, y.Length);
            gpu.Queue.WriteToBuffer(x, inBuffer, true, null);
            gpu.Kernel.SetMemoryArgument(0, inBuffer);
            gpu.Kernel.SetMemoryArgument(1, outBuffer);
            gpu.Queue.Execute(gpu.Kernel, null, localWorkSize, localWorkSize, null);
            gpu.Queue.Finish();
            var y1 = new Float2[y.Length];
            gpu.Queue.ReadFromBuffer(outBuffer, ref y1, true, null);
            var discrep = y.Zip(y1, (ff1, ff2) => ff2.X != 0.0f ? Math.Abs((ff1.X - ff2.X) / ff2.X) : 0.0).ToArray();
            var maxDiscrep = discrep.Max();
        }

        private static Float2[,] Fft1DExact(Float2[,] x)
        {
            var result = new Float2[x.GetLength(0), n];
            for (int k0 = 0, n0 = x.GetLength(0); k0 < n0; k0++)
            {
                for (var k = 0; k < n; k++)
                {
                    var x_ = 0.0;
                    var y_ = 0.0;
                    for (var i = 0; i < n; i++)
                    {
                        var xI = x[k0, i];
                        var angle = 2.0 * Math.PI * i * k / n;
                        var c = Math.Cos(angle);
                        var s = Math.Sin(angle);
                        x_ += c * xI.X - s * xI.Y;
                        y_ += c * xI.Y + s * xI.X;
                    }
                    result[k0, k] = new Float2(x_, y_);
                }
            }
            return result;
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

        private static T[,] To2DArray<T>(this IEnumerable<T> source, int nRows)
        {
            var enumerable = source as T[] ?? source.ToArray();
            var len = enumerable.Length;
            if (len % nRows != 0)
                throw new ArgumentException("nRows");
            var nCols = len / nRows;
            var res = new T[nRows, nCols];
            for (var i = 0; i < nRows; i++)
            {
                for (var j = 0; j < nCols; j++)
                {
                    res[i, j] = enumerable[i * nCols + j];
                }
            }
            return res;
        }
    }
}