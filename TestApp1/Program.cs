using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Cloo;

namespace TestApp1
{
    static class Program
    {
        static void Main(string[] args)
        {
            var device =
                ComputePlatform.Platforms.SelectMany(p => p.Devices)
                    .FirstOrDefault(d => d.Type == ComputeDeviceTypes.Gpu);
            if(device == null)
                return;
            var context = new ComputeContext(new List<ComputeDevice> {device},
                new ComputeContextPropertyList(device.Platform), null, IntPtr.Zero);
            var queue = new ComputeCommandQueue(context, device, ComputeCommandQueueFlags.None);            
            var program = new ComputeProgram(context, new StreamReader(@"CL\kernel1.cl").ReadToEnd());
            try
            {
                program.Build(null, null, null, IntPtr.Zero);
            }
            catch (Exception)
            {

                throw new Exception(program.GetBuildLog(device));
            }
            var kernel = program.CreateKernel("helloWorld");


            const int n = 100;
            var fullCos = new float[n];
            var nativeCos = new float[n];
            var doubleCos = Enumerable.Range(0, n).Select(i => Math.Cos(Math.PI * 2.0 / n * i)).ToArray();
            var fullCosBuffer = new ComputeBuffer<float>(context,
                ComputeMemoryFlags.WriteOnly | ComputeMemoryFlags.UseHostPointer, fullCos);
            var nativeCosBuffer = new ComputeBuffer<float>(context,
                ComputeMemoryFlags.WriteOnly | ComputeMemoryFlags.UseHostPointer, nativeCos);

            kernel.SetMemoryArgument(0, fullCosBuffer); // set the integer array
            kernel.SetMemoryArgument(1, nativeCosBuffer); // set the integer array

            queue.Execute(kernel, null, new long[] {n}, new long[] {n}, null);
            queue.ReadFromBuffer(fullCosBuffer, ref fullCos, true, null);
            queue.ReadFromBuffer(nativeCosBuffer, ref nativeCos, true, null);
            // wait for completion

            var diff = fullCos.Zip(nativeCos, (f, f1) => f1 != 0.0 ? (f - f1) / f1 : 0.0f).ToArray();


            queue.Finish();
        }
    }
}
