using System;

namespace TestApp1
{
    public struct Float2
    {
        public readonly float X;
        public readonly float Y;

        public Float2(float x, float y = 0.0f)
        {
            X = x;
            Y = y;
        }

        public Float2(double x, double y = 0.0)
            : this((float) x, (float) y)
        {
        }

        public Float2 Conj { get { return new Float2(X, -Y); } }
        public float Module2 { get { return X * X + Y * Y; } }
        public float Module { get { return (float) Math.Sqrt(Module2); } }
        public Float2 Inv { get { return Conj / Module2; } }

        public static Float2 operator +(Float2 _1, Float2 _2)
        {
            return new Float2(_1.X + _2.X, _1.Y + _2.Y);
        }

        public static Float2 operator -(Float2 _1, Float2 _2)
        {
            return new Float2(_1.X - _2.X, _1.Y - _2.Y);
        }

        public static Float2 operator *(float _1, Float2 _2)
        {
            return new Float2(_1 * _2.X, _1 * _2.Y);
        }

        public static Float2 operator *(Float2 _2, float _1)
        {
            return new Float2(_1 * _2.X, _1 * _2.Y);
        }

        public static Float2 operator /(Float2 _1, float _2)
        {
            return new Float2(_1.X / _2, _1.Y / _2);
        }

        public static Float2 operator *(Float2 _1, Float2 _2)
        {
            return new Float2(_1.X * _2.X - _1.Y * _2.Y, _1.Y * _2.X + _1.X * _2.Y);
        }

        public static Float2 FromPolar(float module, float argument)
        {
            return new Float2((float) (module * Math.Cos(argument)), (float) (module * Math.Sin(argument)));
        }

        public static Float2 operator /(Float2 _1, Float2 _2)
        {
            return _1 * _2.Inv;
        }
    }
}