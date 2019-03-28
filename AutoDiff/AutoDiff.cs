using System;
using System.Collections.Generic;

namespace AD
{
    /// <summary>
    /// A context contains the values and gradients that have already been calculated, so they can be reused later without having to calculate them again.
    /// </summary>
    public class Context
    {
        public Dictionary<Diffable, float> values;
        public Dictionary<Diffable, Dictionary<Variable, float>> grads;

        public bool ContainsValue(Diffable diffable)
        {
            return values.ContainsKey(diffable);
        }

        public bool ContainsGrads(Diffable diffable)
        {
            return grads.ContainsKey(diffable);
        }

        public void Clear()
        {
            values.Clear();
            grads.Clear();
        }

        public Context()
        {
            values = new Dictionary<Diffable, float>();
            grads = new Dictionary<Diffable, Dictionary<Variable, float>>();
        }
    }

    /// <summary>
    /// Base class for differentiable scalars
    /// </summary>
    public abstract class Diffable
    {

        public abstract float Value(Context context);
        public abstract Dictionary<Variable, float> Grads(Context context);

        public static implicit operator Diffable(float value)
        {
            return new Constant(value);
        }

        public static Symbolic operator +(Diffable a, Diffable b)
        {
            return new Symbolic(
                (Context context) =>
                {
                    return a.Value(context) + b.Value(context);
                },
                (Context context) =>
                {
                    var grads = a.Grads(context);
                    foreach (var grad in b.Grads(context))
                    {
                        if (grads.ContainsKey(grad.Key))
                        {
                            grads[grad.Key] += grad.Value;
                        }
                        else
                        {
                            grads.Add(grad.Key, grad.Value);
                        }
                    }
                    return grads;
                }
            );
        }

        public static Symbolic operator -(Diffable a)
        {
            return new Symbolic(
                (Context context) =>
                {
                    return -a.Value(context);
                },
                (Context context) =>
                {
                    var grads = new Dictionary<Variable, float>();
                    foreach (var grad in a.Grads(context))
                    {
                        grads[grad.Key] = -grad.Value;
                    }
                    return grads;
                }
            );
        }

        public static Symbolic operator -(Diffable a, Diffable b)
        {
            return a + (-b);
        }

        public static Symbolic operator *(Diffable a, Diffable b)
        {
            return new Symbolic(
                (Context context) =>
                {
                    return a.Value(context) * b.Value(context);
                },
                (Context context) =>
                {
                    var grads = new Dictionary<Variable, float>();
                    foreach (var grad in a.Grads(context))
                    {
                        grads.Add(grad.Key, grad.Value * b.Value(context));
                    }
                    foreach (var grad in b.Grads(context))
                    {
                        if (grads.ContainsKey(grad.Key))
                        {
                            grads[grad.Key] += grad.Value * a.Value(context);
                        }
                        else
                        {
                            grads.Add(grad.Key, grad.Value * a.Value(context));
                        }
                    }
                    return grads;
                }
            );
        }

        public static Symbolic Reciprocal(Diffable x)
        {
            return new Symbolic(
                (Context context) =>
                {
                    return 1 / x.Value(context);
                },
                (Context context) =>
                {
                    var grads = new Dictionary<Variable, float>();
                    var negoneoverx = -1 / (x.Value(context) * x.Value(context));
                    foreach (var grad in x.Grads(context))
                    {
                        grads.Add(grad.Key, grad.Value * negoneoverx);
                    }
                    return grads;
                }
            );
        }

        public static Symbolic operator /(Diffable a, Diffable b)
        {
            return a * Reciprocal(b);
        }

        public Symbolic Square()
        {
            return this * this;
        }

    }

            public class Variable : Diffable
        {
            public float value;
            public override float Value(Context context)
            {
                return value;
            }
            public override Dictionary<Variable, float> Grads(Context context)
            {
                return new Dictionary<Variable, float>()
                {
                    { this, 1 }
                };
            }
            public Variable(float value)
            {
                this.value = value;
            }
            public static explicit operator Variable(float value)
            {
                return new Variable(value);
            }
        }

        public class Constant : Diffable
        {
            public float value;
            public override float Value(Context context)
            {
                return value;
            }
            public override Dictionary<Variable, float> Grads(Context context)
            {
                return new Dictionary<Variable, float>();
            }
            public Constant(float value)
            {
                this.value = value;
            }
            public static implicit operator Constant(float value)
            {
                return new Constant(value);
            }
        }

    /// <summary>
    /// Symbolics are containers for mathematical functions that can be differantiated
    /// </summary>
    public class Symbolic : Diffable
    {

        private Func<Context, float> value;
        private Func<Context, Dictionary<Variable, float>> grads;

        public override float Value(Context context)
        {
            if (context.values.TryGetValue(this, out var val))
            {
                return val;
            }
            else
            {
                val = value(context);
                context.values.Add(this, val);
                return val;
            }
        }

        public override Dictionary<Variable, float> Grads(Context context)
        {
            if (context.grads.TryGetValue(this, out var val))
            {
                return val;
            }
            else
            {
                val = grads(context);
                context.grads.Add(this, val);
                return val;
            }
        }

        public Symbolic(Func<Context, float> value, Func<Context, Dictionary<Variable, float>> grads)
        {
            this.value = value;
            this.grads = grads;
        }
    }
}
