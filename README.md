# PINNs

This directory contains multiple methods to calculate the minimizer of the functional given by

$$L_\varepsilon(u)= \frac{1}{2}\lVert -\varepsilon u''+u'-1\rVert_{L^2(0,1)}^2+\frac{1}{4}u(0)^2+\frac{1}{4}u(1)^2$$

We minimize this using $\tanh$-neural networks or using polynomials, parameterized by either the canonical basis functions or the shifted Legendre basis functions.

## Structure of the directory

* `Main`: contains the classes to train a network and visualise the results.
  * `Canonical.py`: contains some functions to calculate the minimizers of $L_\varepsilon$ using the canonical basis. These functions are merely used to illustrate that it is not a good idea to compute the minimizers using this basis.
  * `Legendre.py`: contains a class to obtain the minimizer using the shifted Legendre basis.
  * `Train_v2.py` : contains a function to approximate the minimizer using a neural network and a class which contains all the results.
* `Example_Code.ipynb`: some examples on how to use the code to train the neural network.
* `Poly.ipynb`: some examples on how the `Legendre_solution` class works.
* `Figures.ipynb`: contains the code to generate all used figures.

## How to use the code

The file `Example_Code.ipynb`, shows how to train the network, and the main functions of the class `train_network` are shown.

To train the network, you need to use the following function:

```
train_network(LR: float,
              HIDDEN: int,
              NR_EPOCHS: int,
              h_SAMPLE: float,
              LAMBDA: float,
              PLOT_INTERVAL: int,
              EPSIL: float,
              Initializer=initializeNetwork,
              optim='SGD',
              params=0,
              momentum=0,
              M=-1)
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
* `M`: Bound on parameter size, if `M=-1` the parameters are unbounded.

In the example, the arguments are specified as follows `LR=0.1` etcetera. This is not necessary, but done for clarity.

It returns an object of the class `PINN_solution`, which is explained below.

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
For now, also for more neurons, the results are saved in a 1D array, so we need to untangle it before we make plots.
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
It also has the following functions:
*  `.update(network, loss)`: updates the class after a training step, all values listed above are updated.
*  `.evol_plot(savepath=None)`: makes a plot of the evolution of the parameters, if `savepath` is specified, it saves this plot in the specified path.
*  `.evol_plot_grad(savepath=None)`: makes a plot of the evolution of the gradient of the parameters.
*  `.loss_infl_plots(savepath=None)`: makes a plot of the evolution of the loss and inflection points.
*  `.loss_plot(savepath=None)`: makes a plot of the evolution of the loss.
*  `.sol()`: returns the approximated solution.
*  `.show_sol(savepath=None)`: makes a plot of the approximated solution.

### The class `Legendre_solution`

We can initialize the class using `Legendre_solution(Np,epsil)`. It has the following attributes and functions:
*  `.Np`: degree of the polynomial.
*  `.epsil`: epsilon.
*  `.A`: matrix $A$ corresponding to Legendre system $Ax=b$.
*  `.b`: vector $b$ corresponding to Legendre system $Ax=b$.
*  `.coeffs`: solution $x$ to $Ax=b$.
*  `.Loss`: corresponding loss to minimizer.
*  `.y_coords()`: gives the y coordinates of the solution.
*  `.show_approx()`: plot the approximation
*  `.cond()`: calculate the condition number.
