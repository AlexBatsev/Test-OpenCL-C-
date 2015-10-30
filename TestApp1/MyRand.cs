using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;

namespace TestApp1
{
    public class MyRand
    {
        private readonly double begin;
        private readonly double len;
        private Random rand = new Random();

        public MyRand(double begin, double end)
        {
            this.begin = begin;
            len = end - begin;
        }

        public double Next { get { return begin + rand.NextDouble() * len; } }

        public IEnumerable<double> GetNext(uint n)
        {
            for (int i = 0; i < n; i++)
                yield return Next;
        }

        public IEnumerable<double[]> GetNext(uint n1, uint n2)
        {
            for (int i = 0; i < n1; i++)
                yield return GetNext(n2).ToArray();
        }
    }
}