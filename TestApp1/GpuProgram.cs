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
        public readonly ComputeCommandQueue Queue;

        public ComputeKernel GetKernel(string file, string kernelName)
        {
            var program = new ComputeProgram(Context, new StreamReader(@"CL\" + file).ReadToEnd());
            try
            {
                program.Build(null, null, null, IntPtr.Zero);
            }
            catch (Exception)
            {
                var buildLog = program.GetBuildLog(Device);
                throw new Exception(buildLog);
            }
            System.Diagnostics.Debug.WriteLine(program.GetBuildLog(Device));
            return program.CreateKernel(kernelName);
        }

        public GpuProgram()
        {
            Device =
                ComputePlatform.Platforms.SelectMany(p => p.Devices)
                    .FirstOrDefault(d => d.Type == ComputeDeviceTypes.Gpu);
            if (Device == null)
                return;
            Context = new ComputeContext(new List<ComputeDevice> {Device},
                new ComputeContextPropertyList(Device.Platform), null, IntPtr.Zero);
            Queue = new ComputeCommandQueue(Context, Device, ComputeCommandQueueFlags.None);
        }


        public ComputeBuffer<Float2> CreateBuffer(long n)
        {
            return new ComputeBuffer<Float2>(Context, ComputeMemoryFlags.ReadWrite, n);
        }

        public ComputeImage2D CreateImage2D()
        {
            return new ComputeImage2D(Context, ComputeMemoryFlags.ReadWrite,
                new ComputeImageFormat(ComputeImageChannelOrder.RG, ComputeImageChannelType.Float), 256, 256, 0,
                IntPtr.Zero);
        }

        public void Exec1D(ComputeKernel kernel, long global, long local)
        {
            Queue.Execute(kernel, null, new[] {global}, new[] {local}, null);
        }

        public void Exec2D(ComputeKernel kernel, long global1, long global2, long local1, long local2)
        {
            Queue.Execute(kernel, null, new[] { global1, global2 }, new[] { local1, local2 }, null);
        }
    }
}