using System.Collections.Generic;

namespace AD
{
    public static class Math
    {
        public static Symbolic Exp(Diffable x)
        {
            return new Symbolic(
                (Context context) =>
                {
                    return (float)System.Math.Exp(x.Value(context));
                },
                (Context context) =>
                {
                    var grads = new Dictionary<Variable, float>();
                    var expvalue = (float)System.Math.Exp(x.Value(context));
                    foreach (var grad in x.Grads(context))
                    {
                        grads[grad.Key] = grad.Value * expvalue;
                    }
                    return grads;
                }
            );
        }
        public static Symbolic Log(Diffable x)
        {
            return new Symbolic(
                (Context context) =>
                {
                    return (float)System.Math.Log(x.Value(context));
                },
                (Context context) =>
                {
                    var grads = new Dictionary<Variable, float>();
                    var value = x.Value(context);
                    foreach (var grad in x.Grads(context))
                    {
                        grads[grad.Key] = grad.Value / value;
                    }
                    return grads;
                }
            );
        }
        public static Symbolic Sqrt(Diffable x)
        {
            return new Symbolic(
                (Context context) =>
                {
                    return (float)System.Math.Sqrt(x.Value(context));
                },
                (Context context) =>
                {
                    var grads = new Dictionary<Variable, float>();
                    float twosqrtx = 2 * (float)System.Math.Sqrt(x.Value(context));
                    foreach (var grad in x.Grads(context))
                    {
                        grads[grad.Key] = grad.Value / twosqrtx;
                    }
                    return grads;
                }
            );
        }

        public static Symbolic Pow(Diffable x, Diffable p)
        {
            return Exp(Log(x) * p);
        }

        public static Symbolic Abs(Diffable x)
        {
            return new Symbolic(
                (Context context) =>
                {
                    return System.Math.Abs(x.Value(context));
                },
                (Context context) =>
                {
                    var grads = new Dictionary<Variable, float>();
                    var valuesign = System.Math.Sign(x.Value(context));
                    foreach (var grad in x.Grads(context))
                    {
                        grads[grad.Key] = grad.Value * valuesign;
                    }
                    return grads;
                }
            );
        }

        public static Symbolic Min(Diffable a, Diffable b)
        {
            return new Symbolic(
                (Context context) =>
                {
                    return System.Math.Min(a.Value(context), b.Value(context));
                },
                (Context context) =>
                {
                    var grads = new Dictionary<Variable, float>();

                    bool aLessThanB = (a.Value(context) < b.Value(context));
                    foreach (var grad in a.Grads(context))
                    {
                        if (aLessThanB)
                        {
                            grads.Add(grad.Key, grad.Value);
                        }
                        else
                        {
                            grads.Add(grad.Key, 0);
                        }
                    }
                    foreach (var grad in b.Grads(context))
                    {
                        if (grads.ContainsKey(grad.Key))
                        {
                            if (!aLessThanB)
                            {
                                grads[grad.Key] = grad.Value;
                            }
                        }
                        else
                        {
                            if (aLessThanB)
                            {
                                grads.Add(grad.Key, 0);
                            }
                            else
                            {
                                grads.Add(grad.Key, grad.Value);
                            }
                        }
                    }
                    return grads;
                }
            );
        }

        public static Symbolic Max(Diffable a, Diffable b)
        {
            return new Symbolic(
                (Context context) =>
                {
                    return System.Math.Max(a.Value(context), b.Value(context));
                },
                (Context context) =>
                {
                    var grads = new Dictionary<Variable, float>();

                    bool aGreaterThanB = (a.Value(context) > b.Value(context));
                    foreach (var grad in a.Grads(context))
                    {
                        if (aGreaterThanB)
                        {
                            grads.Add(grad.Key, grad.Value);
                        }
                        else
                        {
                            grads.Add(grad.Key, 0);
                        }
                    }
                    foreach (var grad in b.Grads(context))
                    {
                        if (grads.ContainsKey(grad.Key))
                        {
                            if (!aGreaterThanB)
                            {
                                grads[grad.Key] = grad.Value;
                            }
                        }
                        else
                        {
                            if (aGreaterThanB)
                            {
                                grads.Add(grad.Key, 0);
                            }
                            else
                            {
                                grads.Add(grad.Key, grad.Value);
                            }
                        }
                    }
                    return grads;
                }
            );
        }

        public static Symbolic Square(Diffable x)
        {
            return x * x;
        }
    }
}
