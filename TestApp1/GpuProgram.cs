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
            System.Diagnostics.Debug.WriteLine(Program.GetBuildLog(Device));
            Kernel = Program.CreateKernel(kernelName);
        }
    }
}