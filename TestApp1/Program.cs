using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Cloo;

namespace TestApp1
{
    public class GpuProgram
    {
        public readonly ComputeContext Context;
        public readonly ComputeDevice Device;
        public readonly ComputeKernel Kernel;
        private readonly ComputeProgram Program;
        public readonly ComputeCommandQueue Queue;

        public GpuProgram(string file, string kernelName)
        {
            Device =
                ComputePlatform.Platforms.SelectMany(p => p.Devices)
                    .FirstOrDefault(d => d.Type == ComputeDeviceTypes.Gpu);
            if (Device == null)
                return;
            Context = new ComputeContext(new List<ComputeDevice> {Device},
                new ComputeContextPropertyList(Device.Platform), null, IntPtr.Zero);
            Queue = new ComputeCommandQueue(Context, Device, ComputeCommandQueueFlags.None);
            Program = new ComputeProgram(Context, new StreamReader(@"CL\" + file).ReadToEnd());
            try
            {
                Program.Build(null, null, null, IntPtr.Zero);
            }
            catch (Exception)
            {
                throw new Exception(Program.GetBuildLog(Device));
            }
            Kernel = Program.CreateKernel(kernelName);
        }
    }

    internal static class Program
    {
        private static void Main(string[] args)
        {
            var gpu = new GpuProgram("fft256.cl", "fft256");

            const int n = 256;
            const int n2 = n * 2;
            var rnd = new Random();
            var x = Enumerable.Range(0, n2).Select(i => (float)((rnd.NextDouble() - 0.5) * 100.0)).ToArray();
            var y = new float[n2];
            var inBuffer = new ComputeBuffer<float>(gpu.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.UseHostPointer, x);
            var outBuffer = new ComputeBuffer<float>(gpu.Context, ComputeMemoryFlags.ReadWrite, y.Length);
            gpu.Queue.WriteToBuffer(x, inBuffer, true, null);
            gpu.Kernel.SetMemoryArgument(0, inBuffer);
            gpu.Kernel.SetMemoryArgument(1, outBuffer);
            gpu.Queue.Execute(gpu.Kernel, null, new long[] { 64 }, new long[] { 64 }, null);
            gpu.Queue.Finish();
            gpu.Queue.ReadFromBuffer(outBuffer, ref y, true, null);
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