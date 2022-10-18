# Multicategory Vertex Discriminant Analysis (MVDA)

A Julia package for classification using a vertex-valued encoding of data labels.
Work in progress.

## Installation

This package requires Julia 1.7.0 or higher.

The package folder, call it `MVDA`, can live anywhere on your file directory.
However, any Julia session using `MVDA` must use this package's environment.
Let's say we have installed the package at `/myhome/dir/MVDA/`.

1. In a Julia session, type `]` to enter package mode and then enter `activate /myhome/dir/MVDA`. The package mode prompt should change from `(@v1.7) pkg>` to `(MVDA) pkg>`.
2. In package mode, run the command `instantiate`.
3. Enter `Backspace` to return to the Julia REPL.

Example Julia session:

```julia
julia>                                  # starts with default prompt
(@v1.7) pkg>                            # enter package mode with `]`
(@v1.7) pkg> activate /myhome/dir/MVDA/ # specify environment
...
(MVDA) pkg> instantiate
```

## Using the software

Start Julia from the command line with `julia --project=/myhome/dir/MVDA` as a shortcut or simply run `] activate /myhome/dir/MVDA`.
Once in the correct project environment, simply run `using MVDA` to get started.

## Running package tests

In Julia, with the `MVDA` environment activated, enter package mode and then run the `test` command:

```julia
(MVDA) pkg> test # everything should pass!
```

# Examples 
Not all functions in this package are exported, so they must be qualified with `MVDA.<function name>`.
The following examples highlight the most important functions.

## Loading demo datasets

```julia
using MVDA

MVDA.list_datasets()        # lists available demo datasets
df = MVDA.dataset("iris")   # loads the `iris` dataset as a DataFrame
```

## Creating an instance of a MVDA problem

The `MVDAProblem` type accepts data as two arguments: a `Vector` of labels and a `Matrix` of features/predictors/covariates.

<details>
<summary>Click to expand</summary>

```julia
using MVDA

df = MVDA.dataset("iris")
L, X = Vector(df[!,1]), Matrix{Float64}(df[!,2:end]) # targets/labels are always on first column
problem = MVDAProblem(targets, X) # initialize; n samples, p features, c classes

n, p, c = MVDA.probdims(problem) # get dimensions
```

**Checking properties of a model object:**

```julia
problem.X               # n × p data/design matrix
problem.Y               # n × c-1 response embedded in vertex space

problem.intercept       # Boolean value indicating whether to estimate an intercept [default=true]
problem.kernel          # Object representing the choice of kernel [default=nothing]

problem.encoding        # object representing encoding of classes in vertex space
problem.encoding.vertex # list of vertices in the encoding
problem.label2vertex    # associative map: targets/labels to rows of Y

# the fields coeff, coeff_prev, proj, and grad have the subfields `slope` and `intercept`:

problem.coeff.slope     # p × c-1 coefficient matrix
problem.coeff.intercept # c-1 × 1 intercept in vertex space; only used if `intercept = true`

problem.coeff_prev      # same as problem.coeff, but used for Nesterov acceleration

problem.proj            # same as problem.coeff, but stores sparse projection

problem.grad            # same as problem.coeff, but stores gradient with respect to coefficients

# residuals are split into 2 fields: main, dist, and weighted

problem.res.loss        # n × c-1; used to store scaled residuals Y - X*B
problem.res.dist        # p × c-1; used to store residuals P(B) - B
```

</details>

## Fitting a model

The functions `MVDA.fit!` and `MVDA.anneal!` are the workhorses of this package.

* `MVDA.fit!` solves the penalized problem along the annealing path; that is, as `ρ` approaches infinity. This handles *outer* iterations.
* `MVDA.anneal!` solves the penalized problem for a particular value of `ρ`. This handles *inner* iterations of a proximal distance algorithm.

The default annealing schedule is $\rho(t) = \min\{\rho_{\max}, \rho_{0} 1.2^{t}\}$ at outer iteration $t$, with $\rho_{0} = 1$ and $\rho_{\max} = 10^{8}$.

Optional arguments are specified with `<option>=<value>` pairs; examples highlighted below.

**Note!!!** The first call to `MVDA.fit!` or `MVDA.anneal!` will incur a precompilation step.
Subsequent calls to the same function will typically run faster than the initial call, provided similar *types* of arguments are used.
The timing results for `@time` assume precompilation has already taken place.

### Examples with `MVDA.fit!`: No sparsity

<details>
<summary>Click to expand</summary>

```julia
using MVDA, Random

# create the problem instance
df = MVDA.dataset("iris")
L, X = Vector(df[!,1]), Matrix{Float64}(df[!,2:end])
problem = MVDAProblem(L, X, kernel=nothing, intercept=true)
n, p, c = MVDA.probdims(problem)

# fit VDA model using SVD-based variant; no sparsity

epsilon = MVDA.maximum_deadzone(problem)    # use maximum radius for non-overlapping deadzones
lambda = 1.0                                # regularization strength

((iters, result), final_rho) = @time MVDA.fit!(MMSVD(), problem, epsilon, lambda,
    maxiter=10^4,   # maximum number of inner iterations (affects convergence for ρ fixed)
    gtol=1e-3,      # control quality of solutions for fixed rho, i.e. |∇f(B)| < 1e-3
    nesterov=10,          # minimum number of steps to take WITHOUT Nesterov accel.
    callback=VerboseCallback(5),    # print convergence information every 5 iterations
)
```


**Output from `VerboseCallback`:** Iterations column indicates the current iteration.

```
iter 	rho     	risk    	loss    	objective	penalty     	|gradient|	distance
   0	0.000e+00	1.795e-02	8.975e-03	8.975e-03	0.000e+00	1.881e-01	0.000e+00
   5	0.000e+00	2.386e-03	1.881e-03	1.881e-03	5.506e-03	7.366e-02	0.000e+00
  10	0.000e+00	1.401e-03	1.578e-03	1.578e-03	7.023e-03	2.768e-02	0.000e+00
  15	0.000e+00	1.045e-03	1.524e-03	1.524e-03	8.011e-03	8.118e-03	0.000e+00
  20	0.000e+00	9.236e-04	1.518e-03	1.518e-03	8.447e-03	1.038e-03	0.000e+00
  0.000497 seconds (193 allocations: 34.953 KiB)
```

**Returned values:**

```
iters               # total number of iterations taken, inner + outer
result.risk         # empirical risk
result.loss         # regularized empirical risk
result.penalty      # Frobenius norm of coefficients, |B|²
result.objective    # 0.5 * (loss + ρ × dist(B,S)²)
result.distance     # distance penalty, dist(B,S)
result.gradient     # gradient norm, |∇f(B)| = |∇g(B∣B)|
```

**Checking accuracy and estimates:**

```
accuracy = MVDA.accuracy(problem, (L,X));
println("Training accuracy is ", accuracy*100, "%.") # should be >80%

problem.coeff.slope     # estimate of coefficients
problem.coeff.intercept # estimate of intercept
```

</details>

### Example with `MVDA.fit!`: Sparsity version

<details>
<summary>Click to expand</summary>

```julia
using MVDA, Random

# create the problem instance
df = MVDA.dataset("iris")
L, X = Vector(df[!,1]), Matrix{Float64}(df[!,2:end])
problem = MVDAProblem(L, X)
n, p, c = MVDA.probdims(problem)

# fit VDA model using SVD-based variant; sparsity=0.25
epsilon = MVDA.maximum_deadzone(problem)    # use maximum radius for non-overlapping deadzones
lambda = 1.0                                # regularization strength
sparsity = 0.25                             # drop 1 feature

((iters, result), final_rho) = @time MVDA.fit!(MMSVD(), problem, epsilon, lambda, sparsity,
    maxrhov=100,    # maximum number of outer iterations (ρ to try)
    maxiter=10^4,   # maximum number of inner iterations (affects convergence for ρ fixed)
    dtol=1e-6,      # control quality of distance squared, i.e. dist(B,S) < 1e-6
    rtol=1e-6,      # check progress made on distance squared on relative scale
    rho_init=1.0,   # initial value for rho
    rho_max=1e8,    # maximum value for rho
    gtol=1e-3,      # control quality of solutions for fixed rho, i.e. |∇f(B)| < 1e-3
    nesterov=10,    # minimum number of steps to take WITHOUT Nesterov accel.
    callback=VerboseCallback(5),    # print convergence information every 5 iterations
)
```

**Output from `VerboseCallback`:** Iterations column indicates the current iteration and rho reflects which outer iteration we are working on. Some values may be suppressed if, for example, a single iteration is taken to solve a subproblem on the annealing path.
Set `VerboseCallback(1)` to see the full history.

```
iter 	rho     	risk    	loss    	objective	penalty     |gradient|	distance
   0	1.000e+00	1.795e-02	8.975e-03	8.975e-03	0.000e+00	1.881e-01	0.000e+00
   5	1.000e+00	2.415e-03	1.911e-03	1.919e-03	5.625e-03	7.310e-02	7.972e-03
  10	1.000e+00	1.426e-03	1.600e-03	1.613e-03	7.096e-03	2.763e-02	1.023e-02
  15	1.000e+00	1.061e-03	1.541e-03	1.558e-03	8.080e-03	7.937e-03	1.164e-02
  20	1.000e+00	9.365e-04	1.533e-03	1.552e-03	8.516e-03	9.787e-04	1.227e-02
  0.002547 seconds (920 allocations: 52.188 KiB)
```

**Returned values:**

```
iters               # total number of iterations taken, inner + outer
result.risk         # empirical risk
result.loss         # regularized empirical risk
result.penalty      # Frobenius norm of coefficients, |B|²
result.objective    # 0.5 * (loss + ρ × dist(B,S)²)
result.distance     # distance penalty, dist(B,S)
result.gradient     # gradient norm, |∇f(B)| = |∇g(B∣B)|
```

**Checking accuracy and estimates:**

```
accuracy = MVDA.accuracy(problem, (L,X));
println("Training accuracy is ", accuracy*100, "%.") # should be >80%

problem.coeff_proj.slope        # estimate of coefficients
problem.coeff_proj.intercept    # estimate of intercept
```

</details>

### Example with `MVDA.anneal!`

<details>
<summary>Click to expand</summary>

```julia
using MVDA, Random

# create the problem instance
df = MVDA.dataset("iris")
L, X = Vector(df[!,1]), Matrix{Float64}(df[!,2:end])
problem = MVDAProblem(L, X)
n, p, c = MVDA.probdims(problem)

# solve VDA model using SVD-based variant at particular point in annealing path
epsilon = MVDA.maximum_deadzone(problem)    # use maximum radius for non-overlapping deadzones
lambda = 1.0                                # regularization strength
sparsity = 0.25                             # drop 1 feature
rho = 1.0                                   # distance penalty strength

(iters, result) = @time MVDA.anneal!(MMSVD(), problem, epsilon, lambda, sparsity, rho,
    maxiter=10^4,   # maximum number of inner iterations (affects convergence for ρ fixed)
    gtol=1e-3,      # control quality of solutions for fixed rho, i.e. |∇f(B)| < 1e-3
    nesterov=10,    # minimum number of steps to take WITHOUT Nesterov accel.
    callback=VerboseCallback(1),    # print convergence information every iteration
)
```

**Output from `VerboseCallback`:**

```
   1	1.000e+00	7.251e-03	3.920e-03	3.928e-03	2.358e-03	2.156e-01	8.056e-03
   2	1.000e+00	4.862e-03	2.909e-03	2.915e-03	3.821e-03	1.613e-01	6.995e-03
   3	1.000e+00	3.628e-03	2.394e-03	2.400e-03	4.639e-03	1.202e-01	6.953e-03
   4	1.000e+00	2.893e-03	2.095e-03	2.102e-03	5.190e-03	9.256e-02	7.409e-03
   5	1.000e+00	2.415e-03	1.911e-03	1.919e-03	5.625e-03	7.310e-02	7.972e-03
   6	1.000e+00	2.085e-03	1.792e-03	1.801e-03	5.997e-03	5.873e-02	8.517e-03
   7	1.000e+00	1.846e-03	1.714e-03	1.724e-03	6.327e-03	4.791e-02	9.041e-03
   8	1.000e+00	1.667e-03	1.661e-03	1.672e-03	6.618e-03	3.956e-02	9.501e-03
   9	1.000e+00	1.531e-03	1.625e-03	1.637e-03	6.874e-03	3.293e-02	9.891e-03
  10	1.000e+00	1.426e-03	1.600e-03	1.613e-03	7.096e-03	2.763e-02	1.023e-02
  11	1.000e+00	1.343e-03	1.582e-03	1.596e-03	7.288e-03	2.334e-02	1.051e-02
  12	1.000e+00	1.260e-03	1.567e-03	1.582e-03	7.496e-03	1.896e-02	1.081e-02
  13	1.000e+00	1.184e-03	1.555e-03	1.570e-03	7.706e-03	1.481e-02	1.111e-02
  14	1.000e+00	1.117e-03	1.546e-03	1.563e-03	7.904e-03	1.110e-02	1.139e-02
  15	1.000e+00	1.061e-03	1.541e-03	1.558e-03	8.080e-03	7.937e-03	1.164e-02
  16	1.000e+00	1.017e-03	1.537e-03	1.555e-03	8.228e-03	5.373e-03	1.184e-02
  17	1.000e+00	9.840e-04	1.535e-03	1.553e-03	8.344e-03	3.434e-03	1.201e-02
  18	1.000e+00	9.605e-04	1.534e-03	1.552e-03	8.429e-03	2.087e-03	1.213e-02
  19	1.000e+00	9.452e-04	1.533e-03	1.552e-03	8.485e-03	1.297e-03	1.222e-02
  20	1.000e+00	9.365e-04	1.533e-03	1.552e-03	8.516e-03	9.787e-04	1.227e-02
  0.000624 seconds (750 allocations: 89.812 KiB)
```

**Returned values:**

```
iters               # total number of iterations taken, inner + outer
result.risk         # empirical risk
result.loss         # regularized empirical risk
result.penalty      # Frobenius norm of coefficients, |B|²
result.objective    # 0.5 * (loss + ρ × dist(B,S)²)
result.distance     # distance penalty, dist(B,S)
result.gradient     # gradient norm, |∇f(B)| = |∇g(B∣B)|
```

**Checking accuracy and estimates:**

```
accuracy = MVDA.accuracy(problem, (L,X));
println("Training accuracy is ", accuracy*100, "%.")

problem.coeff_proj.slope        # estimate of coefficients
problem.coeff_proj.intercept    # estimate of intercept
```

</details>

## Cross-Validation

The function `MVDA.cv` can be used to tune `epsilon`, `lambda`, `gamma` (nonlinear models only), and `sparsity` via $k$-fold cross-validation.
It supports the same optional arguments as `MVDA.fit`.

Cross validation is split into two phases

1. The *tuning phase* selects optimal values for `epsilon`, `lambda`, and `gamma` (when applicable) using $k$-fold cross validation.
2. The *path phase* steps through different sparsity values to select a sparse model.

After cross validation, we fit a final sparse model using the selected hyperparameters.
For comparison purposes we also fit a reduced model including only those features selected by the sparse model.

<details>
<summary> Example of k-fold cross validation</summary>

```julia
using MVDA, Random, Statistics, StatsBase, MLDataUtils, StableRNGs

# create the problem instance
df = MVDA.dataset("iris")
L, X = Vector(df[!,1]), Matrix{Float64}(df[!,2:end])
problem = MVDAProblem(L, X, kernel=nothing, intercept=true)
n, p, c = MVDA.probdims(problem)

e_grid = range(1e-2, MVDA.maximum_deadzone(problem), length=11) # epsilon grid
l_grid = range(1e-3, 1e3, length=7)                             # lambda grid
g_grid = [0.0]                                                  # gamma grid (not used here)
s_grid = [1-k/p for k in p:-1:0]                                # sparsity grid

result = @time MVDA.cv(MMSVD(), problem, (e_grid, l_grid, g_grid, s_grid),
    data=MVDA.split_dataset(problem, 0.8),  # indicate training data; default to 80% training
    nfolds=3,                               # number of cross validation folds
    scoref=MVDA.DEFAULT_SCORE_FUNCTION,     # function to evaluate fitted model
    by=:validation,                         # indicates which part of data is used to select a model
    minimize=false,                         # flag indicates whether to minimize or maximize score
    data_transform=ZScoreTransform,         # data transformation applied to each fold
    rtol=0.0,
    gtol=1e-3,
    dtol=1e-3,
    maxiter=10^6,
    maxrhov=10^2,
);
```

**Output:**
```
result.epsilon  # selected value for epsilon
result.lambda   # selected value for lambda
result.gamma    # selected value for gamma
result.sparsity # selected value for sparsity

result.tune     # cross validation results for tuning phase
result.path     # cross validation results for sparsity path phase
result.fit      # result for fitted sparse model
result.reduced  # result for fitted reduced model
```

</details>

In addition, `MVDA.repeated_cv` can be used to generate multiple replicates of cross validation.
This function splits data into cross validation and test sets. The cross validation set is used to tune hyperparameters and thus further split to training and validation subsets.
We permute the cross validation set in each replicate, but the test set is fixed.

This function accepts the same keyword arguments as `MVDA.cv` and `MVDA.fit!`.

<details>
<summary> Example of repeated k-fold cross validation</summary>

```julia
df = MVDA.dataset("iris")
L, X = Vector(df[!,1]), Matrix{Float64}(df[!,2:end])
problem = MVDAProblem(L, X, kernel=nothing, intercept=true)
n, p, c = MVDA.probdims(problem)

e_grid = 1e1 .^ range(-2, log10(MVDA.maximum_deadzone(problem)), length=11) # epsilon grid
l_grid = 1e1 .^ range(-3, 3, length=7)                          # lambda grid
g_grid = [0.0]                                                  # gamma grid (not used here)
s_grid = [1-k/p for k in p:-1:0]                                # sparsity grid
grids = (e_grid, l_grid, g_grid, s_grid)

MVDA.repeated_cv(MMSVD(), problem, grids;
    dir="iris_test",        # directory to store all results
    title="iris test",      # a label for the cross validation results
    overwrite=true,         # write over previous resutls, if they exist

    at=0.7,                 # propagate CV / Test split
    nreplicates=100,        # number of CV replicates
    nfolds=3,               # propagate number of folds
    rng=StableRNG(1903),    # random number generator for reproducibility

    show_progress=true,
)
```

</details>

### The ordering in sparsity grid (`s_grid` in examples) is important!

- Going from `s=0` to `s=1` traverses model sizes from dense to sparse.
- Going from `s=1` to `s=0` traverses model sizes from sparse to dense.
- Interpretation is lost if `s_grid` is not monotonic.

# Nonlinear Vertex Discriminant Analysis

Our implementation of nonlinear VDA transforms the original classification data using a positive definite kernel.
In this setting model coefficients, `problem.coeff.all`, represent the contribution of individual samples to the classifier's decision boundary.
Specificially, `problem.coeff.all[i, :] .== 0` implies sample `i` has some influence on the decision boundary in vertex-space.
On the other hand, `problem.coeff.all[i, :] .!= 0` implies sample `i` has no influence over the decision boundary.
Users should therefore note that the shape of `problem.coeff.all`, as well as similar fields, is different in the nonlinear setting compared to the linear VDA.

The `MVDAProblem` type supports kernels defined in [KernelFunctions.jl](https://github.com/JuliaGaussianProcesses/KernelFunctions.jl) through the `kernel` keyword.

## Example with `RBFKernel`

<details>
<summary>Click to expand</summary>

```julia
using MVDA, Random, KernelFunctions

# create the problem instance
df = MVDA.dataset("spiral")
L, X = Vector(df[!,1]), Matrix{Float64}(df[!,2:end])
problem = MVDAProblem(L, X, kernel=RBFKernel(), intercept=true)
n, p, c = MVDA.probdims(problem)

# fit VDA model using SVD-based variant

epsilon = MVDA.maximum_deadzone(problem)    # use maximum radius for non-overlapping deadzones
sparsity = 0.5                              # target 50% nonzero weights/coefficients
lambda = 1.0                                # regularization strength

((iters, result), final_rho) = @time MVDA.fit!(MMSVD(), problem, epsilon, lambda, sparsity,
    maxrhov=100,    # maximum number of outer iterations (ρ to try)
    maxiter=10^4,   # maximum number of inner iterations (affects convergence for ρ fixed)
    dtol=1e-3,      # control quality of distance squared, i.e. dist(B,S) < 1e-3
    rtol=1e-6,      # check progress made on distance squared on relative scale
    rho_init=1.0,   # initial value for rho
    rho_max=1e8,    # maximum value for rho
    gtol=1e-3,      # control quality of solutions for fixed rho, i.e. |∇f(B)| < 1e-3
    nesterov=10,    # minimum number of steps to take WITHOUT Nesterov accel.
    callback=VerboseCallback(5),   # print convergence information
);
```

**Output from `VerboseCallback`:**

```
iter 	rho     	risk    	loss    	objective	penalty     	|gradient|	distance
   0	1.000e+00	1.795e-02	8.975e-03	8.975e-03	0.000e+00	3.904e-01	0.000e+00
   5	1.000e+00	5.941e-06	1.174e-05	1.207e-05	1.755e-02	1.313e-03	2.568e-02
0.177673 seconds (541 allocations: 53.616 MiB, 2.25% gc time)
```

**Returned values:**

```
iters               # total number of iterations taken, inner + outer
result.risk         # empirical risk
result.loss         # regularized empirical risk
result.penalty      # Frobenius norm of coefficients, |B|²
result.objective    # 0.5 * (loss + ρ × dist(B,S)²)
result.distance     # distance penalty, dist(B,S)
result.gradient     # gradient norm, |∇f(B)| = |∇g(B∣B)|
```

**Checking accuracy and estimates:**

```
accuracy = MVDA.accuracy(problem, (L,X));
println("Training accuracy is ", accuracy*100, "%.")

problem.coeff_proj.slope        # estimate of coefficients
problem.coeff_proj.intercept    # estimate of intercept
```

</details>
