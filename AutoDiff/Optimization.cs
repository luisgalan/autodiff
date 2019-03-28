using System;

namespace AD.Optimization
{
    /// <summary>
    /// Base class for all optimizers
    /// </summary>
    public abstract class Optimizer
    {
        public Diffable loss;

        public abstract void Minimize(Context context);

        public Optimizer(Diffable loss)
        {
            this.loss = loss;
        }

        protected static void UpdateVars(Diffable loss, Action<Variable, float> updater, Context context)
        {
            foreach (var grad in loss.Grads(context))
            {
                updater(grad.Key, grad.Value);
            }
        }
    }

    /// <summary>
    /// Basic gradient descent optimizer
    /// </summary>
    public class GradientDescent : Optimizer
    {
        public override void Minimize(Context context)
        {
            UpdateVars(loss, (Variable variable, float gradient) =>
            {
                variable.value -= learnRate * gradient;
            }, context);
        }

        public float learnRate;

        public GradientDescent(Diffable loss, float learnRate = 0.001f) : base(loss)
        {
            this.learnRate = learnRate;
        }
    }
}
