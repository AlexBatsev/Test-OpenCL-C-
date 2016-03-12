namespace TestApp1
{
    internal struct Interval
    {
        private static float ToMainPeriod(float val)
        {
            const float period = 256.0f;
            while (val < 0)
                val += period;
            while (val > period)
                val -= period;
            return val;
        }

        public Interval(float x)
        {
            x = ToMainPeriod(x);
            Index = (int)x;
            Delta = x - Index;
        }

        public float Combine(float v0, float v1) => v0 * Coeff0 + v1 * Coeff1;
        public Float2 Combine(Float2 v0, Float2 v1) => v0*Coeff0 + v1*Coeff1;

        public int Index { get; }
        public float Delta { get; }
        public int NextIndex => Index == 255 ? 0 : Index + 1;
        public float Coeff0 => 1.0f - Delta;
        public float Coeff1 => Delta;
    }
}