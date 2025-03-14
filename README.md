# martijn-kessels-mep

## Structure of the directory

All Python code is in the folder `CODE`. The code in the folder `CODE/Old_code` might not work, due to the restructering of the folders. The code in `Code/Experiments` should all work, which also includes a code example. The folder `CODE/Experiments/Main` includes some modules that are needed to execute the code. Most resulting figures are saved somewhere in the folder `Latex`, which also includes all tex files. Data is saved in the folder `Saved_Data`.

### Experiments

* `Main`: contains the classes to train a network and visualise the results
* `Example_Code.ipynb`: an example code, explaine in the section **How to use the code**;
* `GradientFlow.ipynb`: analyse a system of one neuron using gradient flow to minimize, instead of neural networks;
* `Hessian.ipynb`: calculate all derivatives, semi-analytically. In fact the Hessian is not calculated;
* `Minimize_a.ipynb`: here we minimize over $a$ first, and then over $u_0$ and $u_1$. Related to the coordinate transform;
* `Polynomial.ipynb`: fits the best polynomials of degree $n$, also contains a part where the theoretical guess using perturbation theory is analyzed.

## How to use the code

In the folder `Code/Experiments/`, the file `Example_Code.ipynb`, is an example of how to use the code is added, and it's main functions are shown.

To train the network, you need to use the following function:

```
train_network(LR: float, HIDDEN: int, NR_EPOCHS: int, h_SAMPLE: float, LAMBDA: float, PLOT_INTERVAL: int, EPSIL: float, Initializer=initializeNetwork, optim='SGD', params=0)
```
It has the following arguments:
* `LR`: Learning rate;
* `HIDDEN`: The number of neurons
* `NR_EPOCHS`: The number of epochs we will train for
* `h_SAMPLE`: The number of points we use to discretize the integral (I am aware N_SAMPLE would be a better name)
* `LAMBDA`: Parameter to balance interior and boundary term.
* `PLOT_INTERVAL`: After how many epochs we want to show how the current approximation looks like. By setting to -1, no output plot is given.
* `EPSIL`: epsilon.
* `Initializer`: How we want to initialize the network, by default pytorch standard.
* `Optimizer`: By default SGD.
* `params`: initial parameters for different initializer than default. If not specified we set it equal to zero and the code will neglect it.
* `momentum`: Momentum, by default 0.

In the example, the arguments are specified as follows `LR=0.1` etcetera. This is not necessary, but done for clarity.

It returns an object of the class `PINN_solution`, which will be explained below.

### Initializer
By default pytorch standard. If we set it to `initializeNetwork_guess`, we need to specify initial values. We should input them like `[w,a,b]`, where the parameters look as follows:
```
w = torch.tensor([[1.],
                  [2.],
                  [3.]]).to(device)
a = torch.tensor([[1.,2.,3.]).to(device)
b = torch.tensor([1.,2.,3.]).to(device)
```
**Warning! It might be necessary to add statements like `x = x.float()` after the specification of w, a and b**. This is because numpy and pytorch do not necessary use same precision variables. This might give errors in pytorch that data has a different dtype. For example using integers will give problems.

### Optimizer
By default SGD. Also possible to use LBFGS.

### The class `PINN_solution`
Every step, the loss, inflection points and parameter values are saved.
For now, also for more neurons, the results are saved in a 1D array, so we need to untangle it before we make plots. I might change this later.
You can acces the lists as follows:

```
Result = train_network(...)
Result.w_list
      .a_list
      .b_list
      .w_grad_list
      .a_grad_list
      .b_grad_list
      .infl_list
      .loss_list
      .loss
      .hidden
```
With `.evol_plot()`, you get a picture of the evolution of the parameters w,a,b, and with `.evol_plot_grad`, you get a figure but then of the derivatives. With `.loss_infl_plots()`. The loss and the inflection points are shown.