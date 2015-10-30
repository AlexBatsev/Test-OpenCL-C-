using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using Cloo;

namespace TestApp1
{
    internal static class Program
    {
        private const int n = 256;
        private const int nRows = 256;
        private static readonly int workSize = 64;
        private static readonly long[] localWorkSize = {workSize, 1};
        private static readonly GpuProgram gpu = new GpuProgram();

        private static void Main()
        {
            TestFft2D_noTranspose();
        }

        private static void TestFft2D_noTranspose()
        {
            var fft1D1 = gpu.GetKernel("fft256_and_transpose.cl", "fft256_and_transpose");
            var fft1D2 = gpu.GetKernel("fft256_and_transpose.cl", "fft256_and_transpose");

            var x = Generate(n * n);
            var yExact = Fft2DCpuNoTranspose(x);
            var length = x.Length;

            var buffer0 = gpu.CreateBuffer(length);
            var buffer1 = gpu.CreateBuffer(length);
            var buffer2 = gpu.CreateBuffer(length);

            gpu.Queue.WriteToBuffer(x, buffer0, true, null);
            fft1D1.SetMemoryArgument(0, buffer1);
            fft1D1.SetMemoryArgument(1, buffer2);
            fft1D2.SetMemoryArgument(0, buffer2);
            fft1D2.SetMemoryArgument(1, buffer1);

            var clock = new Stopwatch();
            clock.Start();
            for (int i = 0; i < 10; i++)
            {
                gpu.Queue.CopyBuffer(buffer0, buffer1, null);
                gpu.Exec2D(fft1D1, workSize, nRows, workSize, 1);
                gpu.Exec2D(fft1D2, workSize, nRows, workSize, 1);
            }
            clock.Stop();
            Console.WriteLine("Fft time: {0} ms", clock.ElapsedMilliseconds);

            var yGpu = new Float2[length];
            gpu.Queue.ReadFromBuffer(buffer1, ref yGpu, true, null);
            var discrep = CalcDiscrepancy(yExact, yGpu);
            var maxDiscrep = discrep.OrderBy(arg => arg.Discrepancy).Last().Discrepancy;
        }

        private static void TestFft2D()
        {
            var fft1D = gpu.GetKernel("fft256.cl", "fft256");
            var transpose = gpu.GetKernel("transpose.cl", "transpose");

            var x = Generate(n * n);
            var yExact = Fft2DCpu(x);
            var length = x.Length;

            var buffer0 = gpu.CreateBuffer(length);
            var buffer1 = gpu.CreateBuffer(length);
            var buffer2 = gpu.CreateBuffer(length);

            gpu.Queue.WriteToBuffer(x, buffer0, true, null);
            fft1D.SetMemoryArgument(0, buffer1);
            fft1D.SetMemoryArgument(1, buffer2);
            transpose.SetMemoryArgument(0, buffer2);
            transpose.SetMemoryArgument(1, buffer1);

            var clock = new Stopwatch();
            clock.Start();
            for (int i = 0; i < 10000; i++)
            {
                gpu.Queue.CopyBuffer(buffer0, buffer1, null);
                gpu.Exec2D(fft1D, workSize, nRows, workSize, 1);
                gpu.Exec2D(transpose, n, n, 16, 16);
                gpu.Exec2D(fft1D, workSize, nRows, workSize, 1);
            }
            clock.Stop();
            Console.WriteLine("Fft time: {0} ms", clock.ElapsedMilliseconds);

            var yGpu = new Float2[length];
            gpu.Queue.ReadFromBuffer(buffer2, ref yGpu, true, null);
            var discrep = CalcDiscrepancy(yExact, yGpu);
            var maxDiscrep = discrep.OrderBy(arg => arg.Discrepancy).Last().Discrepancy;
        }

        private static Float2[] Generate(uint nElms)
        {
            return new MyRand(-100.0, 100.0).GetNext(nElms, 2).Select(d => new Float2(d[0], d[1])).ToArray();
        }

        private static void TestTranspose()
        {
            var kernel = gpu.GetKernel("transpose.cl", "transpose");
            var x = Generate(n * n);
            var yExact = Transpose(x);

            var length = x.Length;

            var inBuffer = gpu.CreateBuffer(length);
            var outBuffer = gpu.CreateBuffer(length);

            gpu.Queue.WriteToBuffer(x, inBuffer, true, null);
            kernel.SetMemoryArgument(0, inBuffer);
            kernel.SetMemoryArgument(1, outBuffer);
            gpu.Exec2D(kernel, n, n, 16, 16);
            var y = new Float2[length];
            gpu.Queue.ReadFromBuffer(outBuffer, ref y, true, null);
            var maxDisc = CalcDiscrepancy(yExact, y).Select(diff => diff.Discrepancy).Max();
        }

        private static void Fft1D()
        {
            var kernel = gpu.GetKernel("fft256.cl", "fft256");

            var x = Generate(nRows * n);
            var yExact = Fft1DExact(x);
            var length = x.Length;

            var inBuffer = new ComputeBuffer<Float2>(gpu.Context, ComputeMemoryFlags.ReadOnly, length);
            var outBuffer = new ComputeBuffer<Float2>(gpu.Context, ComputeMemoryFlags.ReadWrite, length);
            gpu.Queue.WriteToBuffer(x, inBuffer, true, null);
            kernel.SetMemoryArgument(0, inBuffer);
            kernel.SetMemoryArgument(1, outBuffer);
            for (var i = 0; i < 10; i++)
            {
                gpu.Queue.Execute(kernel, null, new[] {workSize, (long) nRows}, localWorkSize, null);
            }
            //gpu.Queue.Execute(kernel, null, new[] {workSize, (long) nRows}, localWorkSize, null);
            var y = new Float2[length];
            gpu.Queue.ReadFromBuffer(outBuffer, ref y, true, null);
            var discrep = CalcDiscrepancy(yExact, y);
            var maxDiscrep = discrep.OrderBy(arg => arg.Discrepancy).Last();
        }

        private static Float2Diff[] CalcDiscrepancy(Float2[] y1, Float2[] y2)
        {
            return y1.Zip(y2, (f1, f2) => new Float2Diff(f1, f2)).ToArray();
        }

        private static void Fft1DFromFile()
        {
            var kernel = gpu.GetKernel("fft256.cl", "fft256");

            var xy =
                File.ReadAllText(@"..\..\1D.txt")
                    .Split(new[] {"\n"}, StringSplitOptions.None)
                    .Where(s => s.Length > 1)
                    .Select(s => s.Split(new[] {"\t"}, StringSplitOptions.None).Select(float.Parse).ToArray())
                    .ToArray();

            var x = xy.Select(floats => new Float2(floats[0])).ToArray();
            var y = xy.Select(floats => new Float2(floats[1], floats[2])).ToArray();

            var inBuffer = new ComputeBuffer<Float2>(gpu.Context,
                ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.UseHostPointer, x);
            var outBuffer = new ComputeBuffer<Float2>(gpu.Context, ComputeMemoryFlags.ReadWrite, y.Length);
            gpu.Queue.WriteToBuffer(x, inBuffer, true, null);
            kernel.SetMemoryArgument(0, inBuffer);
            kernel.SetMemoryArgument(1, outBuffer);
            gpu.Queue.Execute(kernel, null, localWorkSize, localWorkSize, null);
            gpu.Queue.Finish();
            var y1 = new Float2[y.Length];
            gpu.Queue.ReadFromBuffer(outBuffer, ref y1, true, null);
            var discrep = y.Zip(y1, (ff1, ff2) => ff2.X != 0.0f ? Math.Abs((ff1.X - ff2.X) / ff2.X) : 0.0).ToArray();
            var maxDiscrep = discrep.Max();
        }

        private static Float2[] Fft2DCpu(Float2[] x)
        {
            return Fft1DExact(Transpose(Fft1DExact(x)));
        }

        private static Float2[] Fft2DCpuNoTranspose(Float2[] x)
        {
            return Transpose(Fft1DExact(Transpose(Fft1DExact(x))));
        }

        private static Float2[] Fft1DExact(Float2[] x)
        {
            Debug.Assert(x.Length % n == 0);
            var nRows = x.Length / n;
            var result = new Float2[x.Length];
            for (var k0 = 0; k0 < nRows; k0++)
            {
                for (var k = 0; k < n; k++)
                {
                    var xOut = 0.0;
                    var yOut = 0.0;
                    for (var i = 0; i < n; i++)
                    {
                        var xI = x[k0 * n + i];
                        var angle = 2.0 * Math.PI * i * k / n;
                        var c = Math.Cos(angle);
                        var s = Math.Sin(angle);
                        xOut += c * xI.X - s * xI.Y;
                        yOut += c * xI.Y + s * xI.X;
                    }
                    result[k0 * n + k] = new Float2(xOut, yOut);
                }
            }
            return result;
        }

        private static T[] Transpose<T>(T[] x)
        {
            var result = new T[x.Length];
            for (var i = 0; i < n; i++)
                for (var j = 0; j < n; j++)
                    result[j * n + i] = x[i * n + j];
            return result;
        }

        private class Float2Diff
        {
            public readonly double Discrepancy;
            public Float2 First;
            public Float2 Second;

            public Float2Diff(Float2 first, Float2 second)
            {
                First = first;
                Second = second;
                Discrepancy = second.Module != 0.0f ? ((first - second) / second).Module : 0.0;
            }
        }
    }
}