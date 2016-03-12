namespace TestApp1
{
    internal class Float2Diff
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