using System;
using System.Threading.Tasks;

namespace AD
{
    public abstract class Vector
    {
        protected abstract Diffable[] _values { get; }

        public int Rank { get { return _values.Length; } }

        public virtual void PrepareValues(Context context) { }
        public virtual void PrepareGrads(Context context) { }

        public SymbolicVector All(Func<Diffable, Symbolic> func)
        {
            return All(func, this);
        }

        public static SymbolicVector All(Func<Diffable, Symbolic> func, Vector arg1)
        {
            Symbolic[] output = new Symbolic[arg1.Rank];
            for (int i = 0; i < arg1.Rank; i++)
            {
                output[i] = func(arg1._values[i]);
            }
            return new SymbolicVector(output);
        }

        public static SymbolicVector All(Func<Diffable, Diffable, Symbolic> func, Vector arg1, Vector arg2)
        {
            if (arg1.Rank != arg2.Rank)
            {
                throw new RankException("arg1 and arg2's ranks do not match");
            }
            Symbolic[] output = new Symbolic[arg1.Rank];
            for (int i = 0; i < arg1.Rank; i++)
            {
                output[i] = func(arg1._values[i], arg2._values[i]);
            }
            return new SymbolicVector(output);
        }

        public static Symbolic ReduceSum(Vector vector)
        {
            if (vector.Rank == 0)
            {
                throw new RankException("can't sum vector with rank of 0");
            }
            else
            {
                Symbolic sum = new Symbolic(vector._values[0].Value, vector._values[0].Grads);
                for (int i = 1; i < vector.Rank; i++)
                {
                    sum += vector._values[i];
                }
                return new Symbolic(
                    (Context context) =>
                    {
                        return sum.Value(context);
                    },
                    (Context context) =>
                    {
                        return sum.Grads(context);
                    }
                );
            }
        }

        public Symbolic ReduceSum()
        {
            return ReduceSum(this);
        }

        public static Symbolic ReduceMean(Vector vector)
        {
            if (vector.Rank == 0)
            {
                throw new RankException("can't get mean of vector with rank of 0");
            }
            else
            {
                Symbolic sum = new Symbolic(vector._values[0].Value, vector._values[0].Grads);
                for (int i = 1; i < vector.Rank; i++)
                {
                    sum += vector._values[i];
                }
                sum /= vector.Rank;
                return new Symbolic(
                    (Context context) =>
                    {
                        return sum.Value(context);
                    },
                    (Context context) =>
                    {
                        return sum.Grads(context);
                    }
                );
            }
        }

        public Symbolic ReduceMean()
        {
            return ReduceMean(this);
        }

        private static SymbolicVector Multiply(Vector vector, Diffable scalar)
        {
            Symbolic[] output = new Symbolic[vector.Rank];

            for (int i = 0; i < vector.Rank; i++)
            {
                output[i] = vector._values[i] * scalar;
            }

            return new SymbolicVector(output);
        }

        public static SymbolicVector operator *(Vector vector, Diffable scalar)
        {
            return Multiply(vector, scalar);
        }

        public static SymbolicVector operator *(Diffable scalar, Vector vector)
        {
            return Multiply(vector, scalar);
        }

        private static SymbolicVector Add(Vector vector, Diffable scalar)
        {
            Symbolic[] output = new Symbolic[vector.Rank];

            for (int i = 0; i < vector.Rank; i++)
            {
                output[i] = vector._values[i] + scalar;
            }

            return new SymbolicVector(output);
        }

        public static SymbolicVector operator +(Vector vector, Diffable scalar)
        {
            return Add(vector, scalar);
        }

        public static SymbolicVector operator +(Diffable scalar, Vector vector)
        {
            return Add(vector, scalar);
        }

        public static SymbolicVector operator -(Vector vector)
        {
            var output = new Symbolic[vector.Rank];

            for (int i = 0; i < vector.Rank; i++)
            {
                output[i] = -vector._values[i];
            }

            return new SymbolicVector(output);
        }

        public static SymbolicVector operator +(Vector a, Vector b)
        {
            return All((Diffable _a, Diffable _b) =>
            {
                return _a + _b;
            }, a, b);
        }

        public static SymbolicVector operator -(Vector a, Vector b)
        {
            return a + (-b);
        }

        public static SymbolicVector Square(Vector x)
        {
            return x.All(Math.Square);
        }

        public SymbolicVector Square()
        {
            return Square(this);
        }

    }

    public class VarVector : Vector
    {
        protected override Diffable[] _values { get { return Values; } }
        public Variable[] Values { get; private set; }

        public VarVector(params Variable[] values)
        {
            Values = values;
        }

        public VarVector(int size, Func<float> initializer)
        {
            Variable[] values = new Variable[size];

            for (int i = 0; i < size; i++)
            {
                values[i].value = initializer();
            }
        }
    }

    public class ConstVector : Vector
    {
        protected override Diffable[] _values { get { return Values; } }
        public Constant[] Values { get; private set; }

        public ConstVector(params Constant[] values)
        {
            Values = values;
        }
    }

    public class SymbolicVector : Vector
    {
        protected override Diffable[] _values { get { return Values; } }
        public Symbolic[] Values { get; private set; }

        public SymbolicVector(params Symbolic[] values)
        {
            Values = values;
        }

        public override void PrepareValues(Context context)
        {
            Parallel.For(0, Rank, (i) =>
            {
                Values[i].Value(context);
            });
        }

        public override void PrepareGrads(Context context)
        {
            Parallel.For(0, Rank, (i) =>
            {
                Values[i].Grads(context);
            });
        }
    }
}
