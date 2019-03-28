# autodiff

A basic C# auto differentiation library that implements the backpropagation algorithm

Sample usage:

```csharp
// generate linearly shaped sample data
List<AD.Constant> _xs = new List<AD.Constant>();
List<AD.Constant> _label_y = new List<AD.Constant>();

float a = -3;
float b = 5;

for (float x = -50; x < 50; x += 0.1f)
{
    _xs.Add(x);
    _label_y.Add(a * x + b);
}

AD.ConstVector label_y = new AD.ConstVector(_label_y.ToArray());
AD.ConstVector xs = new AD.ConstVector(_xs.ToArray());

// define linear model
AD.Variable pred_a = new AD.Variable(1);
AD.Variable pred_b = new AD.Variable(0);

AD.SymbolicVector py = pred_a * xs + pred_b;

// fit a and b to minimize mean square error
AD.Symbolic loss = (py - label_y).Square().ReduceMean();
AD.Optimization.Optimizer optimizer = new AD.Optimization.GradientDescent(loss, 0.001f);

var iterations = 0;
var context = new AD.Context();

while (loss.Value(context) > 1e-5)
{
    context.Clear();
    optimizer.Minimize(context);

    if (iterations % 200 == 0)
    {
        Console.WriteLine("loss: " + loss.Value(context));
    }

    iterations++;
}

Console.WriteLine("a: " + pred_a.value);
Console.WriteLine("b: " + pred_b.value);
Console.WriteLine("loss: " + loss.Value(context));
Console.WriteLine(iterations + " iterations");
Console.ReadKey();
```
