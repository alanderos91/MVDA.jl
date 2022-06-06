# Multicategory Vertex Discriminant Analysis (MVDA)

A Julia package for classification using a vertex-valued encoding of data labels.
Work in progress.

## Installation

This package requires Julia 1.6.0 or higher.

The package folder, call it `MVDA`, can live anywhere on your file directory.
However, any Julia session using `MVDA` must use this package's environment.
Let's say we have installed the package at `/myhome/dir/MVDA/`.

1. In a Julia session, type `]` to enter package mode and then enter `activate /myhome/dir/MVDA`. The package mode prompt should change from `(@v1.6) pkg>` to `(MVDA) pkg>`.
2. In package mode, run the command `resolve` followed by `instantiate`.

Example Julia session:

```julia
julia>                                  # starts with default prompt
(@v1.6) pkg>                            # enter package mode with `]`
(@v1.6) pkg> activate /myhome/dir/MVDA/ # specify environment
...
(MVDA) pkg> resolve
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
targets, X = Vector(df[!,1]), Matrix{Float64}(df[!,2:end]) # targets/labels are always on first column
problem = MVDAProblem(targets, X) # initialize; n samples, p features, c classes

n, p, c = MVDA.probdims(problem) # get dimensions

problem.X               # n × p data/design matrix
problem.Y               # n × c-1 response embedded in vertex space

problem.vertex          # list of vertices in c-1 space representing classes
problem.vertex2label    # associative map: rows of Y to targets/labels
problem.label2vertex    # associative map: targets/labels to rows of Y

# the fields coeff, coeff_prev, proj, and grad have the subfields `all` and `dim`:

problem.coeff.all       # p × c-1 coefficient matrix
problem.coeff.dim[1]    # p coefficient vector along dimension 1 in vertex space

problem.coeff_prev      # same as problem.coeff, but used for Nesterov acceleration

problem.proj.all        # stores projection of coefficient matrix
problem.proj.dim[1]     # stores projections of columns

problem.grad.all        # stores gradient with respect to coefficient matrix
problem.grad.dim[1]     # slice along column 1

# residuals are split into 3 fields: main, dist, and weighted

problem.res.main        # n × c-1; used to store scaled residuals 1/sqrt(n) * (Y - X*B)
problem.res.dist        # p × c-1; used to store residuals (P(B) - B)
problem.res.weighted    # n × c-1; used to store scaled residuals 1/sqrt(n) * (Zₘ - X*B)

problem.res.main.all    # full matrix
problem.res.main.dim[1] # slice along column 1
```

</details>

## Fitting a model

The functions `MVDA.fit` and `MVDA.anneal` are the workhorses of this package.

* `MVDA.fit(algorithm::AbstractMMAlg, problem::MVDAProblem, ϵ::Real, s::Real; kwargs...)` solves the penalized problem along the annealing path; that is, as `ρ` approaches infinity. This handles *outer* iterations.
* `MVDA.anneal(algorithm::AbstractMMAlg, problem::MVDAProblem, ϵ::Real, ρ::Real, s::Real; kwargs...)` solves the penalized problem for a particular value of `ρ`. This handles *inner* iterations of a proximal distance algorithm.

There are also `MVDA.fit!` and `MVDA.anneal!` (called internally by there)
The default annealing schedule is $\rho(t) = \min\{\rho_{\max}, \rho_{0} 1.2^{t}\}$ at outer iteration $t$, with $\rho_{0} = 1$ and $\rho_{\max} = 10^{8}$.

Optional arguments are specified with `<option>=<value>` pairs; examples highlighted below.

The special case with `s=0` requires some care because the problem is ill-defined.
For this case we provide `MVDA.init!` to fit a $\lambda$-regularized model.

### Example with `MVDA.fit`

<details>
<summary>Click to expand</summary>

```julia
using MVDA, Random

# create the problem instance
df = MVDA.dataset("iris")
targets, X = Vector(df[!,1]), Matrix{Float64}(df[!,2:end])
problem = MVDAProblem(targets, X)
n, p, c = MVDA.probdims(problem)

# IMPORTANT: initialize coefficients
randn!(problem.coeff.all)
copyto!(problem.coeff_prev.all, problem.coeff.all)

# fit VDA model using SVD-based variant

ϵ = MVDA.maximal_deadzone(problem)  # use maximum radius for non-overlapping deadzones
sparsity = 0.25                     # drop 1 feature

result = @time MVDA.fit(MMSVD(), problem, ϵ, sparsity,
    nouter=100,     # maximum number of outer iterations (ρ to try)
    ninner=10^4,    # maximum number of inner iterations (affects convergence for ρ fixed)
    dtol=1e-6,      # control quality of distance squared, i.e. dist(B,S)² < 1e-6
    rtol=1e-6,      # check progress made on distance squared on relative scale
    rho_init=1.0,   # initial value for rho
    rho_max=1e8,    # maximum value for rho
    gtol=1e-6,      # control quality of solutions for fixed rho, i.e. |∇f(B)| < 1e-6
    nesterov_threshold=10,  # minimum number of steps to take WITHOUT Nesterov accel.
    verbose=true,   # print convergence information
)

result.iters        # total number of iterations taken, inner + outer
result.loss         # empirical risk
result.objective    # 0.5 * (empirical risk + ρ × dist(B,S)²)
result.distance     # dist(B,S)
result.gradient     # |∇f(B)|² = |∇g(B∣B)|²

accuracy = sum( MVDA.classify(problem, X) .== targets ) / length(targets) * 100;
println("Training accuracy is ", accuracy, "%.") # should be >90%

problem.coeff.all   # estimate of coefficients before projection
problem.proj.all    # estimate of coefficient after projection
```

</details>

### Example with `MVDA.anneal`

<details>
<summary>Click to expand</summary>

```julia
using MVDA, Random

# create the problem instance
df = MVDA.dataset("iris")
targets, X = Vector(df[!,1]), Matrix{Float64}(df[!,2:end])
problem = MVDAProblem(targets, X)
n, p, c = MVDA.probdims(problem)

# IMPORTANT: initialize coefficients
randn!(problem.coeff.all)
copyto!(problem.coeff_prev.all, problem.coeff.all)

# fit VDA model using SVD-based variant

ϵ = MVDA.maximal_deadzone(problem)  # use maximum radius for non-overlapping deadzones
ρ = 1.0                             # usual starting point
sparsity = 0.25                     # drop 1 feature

result = @time MVDA.anneal(MMSVD(), problem, ϵ, ρ, sparsity,
    ninner=10^4,    # maximum number of inner iterations (affects convergence for ρ fixed)
    gtol=1e-6,      # control quality of solutions for fixed rho, i.e. |∇f(B)| < 1e-6
    nesterov_threshold=10,  # minimum number of steps to take WITHOUT Nesterov accel.
    verbose=true,   # print convergence information
)

result.loss         # empirical risk
result.objective    # 0.5 * (empirical risk + ρ × dist(B,S)²)
result.distance     # dist(B,S)
result.gradient     # |∇f(B)|² = |∇g(B∣B)|²

accuracy = sum( MVDA.classify(problem, X) .== targets ) / length(targets) * 100;
println("Training accuracy is ", accuracy, "%.") # should be >90%

problem.coeff.all   # estimate of coefficients before projection
problem.proj.all    # estimate of coefficient after projection
```

</details>

### Example with `MVDA.init!`:

<details>
<summary>Click to expand</summary>

```julia
using MVDA, Random

# create the problem instance
df = MVDA.dataset("iris")
targets, X = Vector(df[!,1]), Matrix{Float64}(df[!,2:end])
problem = MVDAProblem(targets, X)
n, p, c = MVDA.probdims(problem)

# fit VDA model using SVD-based variant
ϵ = MVDA.maximal_deadzone(problem)  # use maximum radius for non-overlapping deadzones
λ = 1e-3

result = @time MVDA.init!(MMSVD(), problem, ϵ, λ,
    maxiter=10^4,   # maximum number of iterations
    gtol=1e-6,      # control quality of solutions for fixed rho, i.e. |∇f(B)| < 1e-6
    nesterov_threshold=10,  # minimum number of steps to take WITHOUT Nesterov accel.
    verbose=true,   # print convergence information
)

result.iters        # total number of iterations taken, inner + outer
result.loss         # empirical risk
result.objective    # 0.5 * (empirical risk + ρ × dist(B,S)²)
result.gradient     # |∇f(B)|² = |∇g(B∣B)|²

accuracy = sum( MVDA.classify(problem, X) .== targets ) / length(targets) * 100;
println("Training accuracy is ", accuracy, "%.") # should be >90%

problem.coeff.all   # estimate of coefficients before projection
problem.proj.all    # estimate of coefficient after projection
```

</details>

## Cross-Validation

The function `MVDA.cv` can be used to tune `ϵ` and `sparsity` via $k$-fold cross-validation.
It supports the same optional arguments as `MVDA.fit`.
The function `MVDA.cv_estimation` is used to carry out repeated cross-validation.

The special case with sparsity `s=0` is handled using a regularized version of the problem.

<details>
<summary> Example of k-fold cross validation</summary>

```julia
using MVDA, Random, Statistics, MLDataUtils, StableRNGs

# create the problem instance
df = MVDA.dataset("zoo")
data = (Vector(df[!,1]), Matrix{Float64}(df[!,2:end])) # store as a Tuple
shuffled_data = shuffleobs(data, obsdim=1, rng=StableRNG(1234))
problem = MVDAProblem(shuffled_data..., intercept=true, kernel=nothing)
n, p, c = MVDA.probdims(problem)

# IMPORTANT: initialize coefficients
fill!(problem.coeff.all, 1/(p+1))
copyto!(problem.coeff_prev.all, problem.coeff.all)

ϵ_grid = range(1e-2, MVDA.maximal_deadzone(problem), length=3)
s_grid = [1-k/p for k in p:-1:0]

result = @time MVDA.cv(MMSVD(), problem, (ϵ_grid, s_grid),
    nfolds=3,       # number of folds
    at=0.9,         # proportion in training set; rest goes to holdout set for testing
    rtol=0.0,
    gtol=1e-3,
    dtol=1e-3,
    ninner=10^6,
    nouter=10^2,
    maxiter=10^4,   # remaining options used only when s=0
    lambda=1e-3,
);

# result contains 4 metrics: time spent fitting a model, training error, validation error, and test error
# each is stored in a field, `time`, `train`, `validation`, `test`, which stores results specific to each fold
result.time[1]    # timing result across (s, ϵ) grid
mean(result.test) # average cross validation error (over folds) across (s, ϵ) grid
```

</details>

<details>
<summary> Example of repeated k-fold cross validation</summary>

```julia
using MVDA, Random, Statistics, MLDataUtils, StableRNGs

# create the problem instance
df = MVDA.dataset("zoo")
data = (Vector(df[!,1]), Matrix{Float64}(df[!,2:end])) # store as a Tuple
shuffled_data = shuffleobs(data, obsdim=1, rng=StableRNG(1234))
problem = MVDAProblem(shuffled_data...)
n, p, c = MVDA.probdims(problem)

# IMPORTANT: initialize coefficients
fill!(problem.coeff.all, 1/(p+1))
copyto!(problem.coeff_prev.all, problem.coeff.all)

ϵ_grid = range(1e-2, MVDA.maximal_deadzone(problem), length=3)
s_grid = [1-k/p for k in p:-1:0]

results = @time MVDA.cv_estimation(MMSVD(), problem, (ϵ_grid, s_grid),
    nreplicates=50, # number of replicates
    nfolds=3,       # number of folds
    at=0.9,         # proportion in training set; rest goes to holdout set for testing
    rtol=0.0,
    gtol=1e-3,
    dtol=1e-3,
    ninner=10^6,
    nouter=10^2,
    maxiter=10^4,   # remaining options used only when s=0
    lambda=1e-3,
);

# The results object contains individual cv results for each replicate.
# The following computes cv validation error across replicates.
avg_cv_error = map(r -> mean(r.validation), results)

# create a multiobjective score for maximizing accuracy, parsimony, and the deadzone.
function score(s_grid, ϵ_grid, mat)
    # ordering matters here!
    [(100-mat[i,j], 100*s_grid[i], ϵ_grid[j]) for i in eachindex(s_grid), j in eachindex(ϵ_grid)]
end

# check optimal model for a particular replicate
maximum(score(s_grid, ϵ_grid, avg_cv_error[1]))

# check optimal model across replicates
acc_opt, s_opt, ϵ_opt = zeros(50), zeros(50), zeros(50)
for (k, rep) in enumerate(avg_cv_error)
    acc_opt[k], s_opt[k], ϵ_opt[k] = maximum(score(s_grid, ϵ_grid, rep))
end

median(acc_opt), median(s_opt), median(ϵ_opt) 
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
targets, X = Vector(df[!,1]), Matrix{Float64}(df[!,2:end])
problem = MVDAProblem(targets, X, kernel=RBFKernel(), intercept=true)
n, p, c = MVDA.probdims(problem)

# IMPORTANT: initialize coefficients
randn!(problem.coeff.all)
copyto!(problem.coeff_prev.all, problem.coeff.all)

# fit VDA model using SVD-based variant

ϵ = MVDA.maximal_deadzone(problem)  # use maximum radius for non-overlapping deadzones
sparsity = 0.5                     # target 50% nonzero weights/coefficients

result = @time MVDA.fit(MMSVD(), problem, ϵ, sparsity,
    nouter=100,     # maximum number of outer iterations (ρ to try)
    ninner=10^4,    # maximum number of inner iterations (affects convergence for ρ fixed)
    dtol=1e-6,      # control quality of distance squared, i.e. dist(B,S)² < 1e-6
    rtol=1e-6,      # check progress made on distance squared on relative scale
    rho_init=1.0,   # initial value for rho
    rho_max=1e8,    # maximum value for rho
    gtol=1e-6,      # control quality of solutions for fixed rho, i.e. |∇f(B)| < 1e-6
    nesterov_threshold=10,  # minimum number of steps to take WITHOUT Nesterov accel.
    verbose=true,   # print convergence information
)

result.iters        # total number of iterations taken, inner + outer
result.loss         # empirical risk
result.objective    # 0.5 * (empirical risk + ρ × dist(B,S)²)
result.distance     # dist(B,S)
result.gradient     # |∇f(B)|² = |∇g(B∣B)|²

accuracy = sum( MVDA.classify(problem, X) .== targets ) / length(targets) * 100;
println("Training accuracy is ", accuracy, "%.") # should be >90%

problem.coeff.all   # estimate of coefficients before projection
problem.proj.all    # estimate of coefficient after projection
```

</details>
