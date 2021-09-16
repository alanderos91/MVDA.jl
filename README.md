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

# Examples 
Not all functions in this package are exported, so they must be qualified with `MVDA.<function name>`.
The following examples highlight the most important functions.

## Loading demo datasets

```julia
using MVDA

MVDA.list_datasets()        # lists available demo datasets
df = MVDA.dataset("iris")   # loads the `iris` dataset as a DataFrame
```

## Creating an instance of the MVDA problem

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

## Fitting a model

The function `fit_MVDA` is the workhorse of this package. It comes in two flavors:

* `fit_MVDA(algorithm, problem, ϵ, sparsity)` solves the penalized problem along the annealing path; that is, as `ρ` approaches infinity. This handles *outer* iterations of a proximal distance algorithm.
* `fit_MVDA(algorithm, problem, ϵ, ρ, sparsity)` solves the penalized problem for a particular value of `ρ`. This handles *inner* iterations of a proximal distance algorithm.

The default annealing schedule is $\rho(t) = \min\{\rho_{\max}, \rho_{0} 1.2^{t}\}$ at outer iteration $t$, with $\rho_{0} = 1$ and $\rho_{\max} = 10^{8}$.

Optional arguments are specified with `<option>=<value>` pairs; examples highlighted below.
```julia
using MVDA, Random

# create the problem instance
df = MVDA.dataset("iris")
targets, X = Vector(df[!,1]), Matrix{Float64}(df[!,2:end])
problem = MVDAProblem(targets, X)
n, p, c = MVDA.probdims(problem)

# IMPORTANT: initialize coefficients
randn!(problem.coeff.all)

# call fit_MVDA with algorithm MMSVD to fit all categories simultaneously

ϵ = 0.5 * sqrt(2*c/(c-1)) # use maximum radius for non-overlapping deadzones
sparsity = 0.0            # no sparsity

result = @time fit_MVDA(MMSVD(), problem, ϵ, sparsity,
    nouter=100,     # maximum number of outer iterations (ρ to try)
    ninner=10^4,    # maximum number of inner iterations (affects convergence for ρ fixed)
    dtol=1e-6,      # control quality of distance squared, i.e. dist(B,S)² < 1e-6
    rtol=1e-6,      # check progress made on distance squared on relative scale
    rho_init=1.0,   # initial value for rho
    rho_max=1e8,    # maximum value for rho
    gtol=1e-6,      # control quality of solutions for fixed row, i.e. ∇f(B) < 1e-6
    nesterov_threshold=10,  # minimum number of steps to take WITHOUT Nesterov accel.
    verbose=true,   # print convergence information
)

result.iters        # total number of iterations taken, inner + outer
result.loss         # empirical risk
result.objective    # 0.5 * (empirical risk + ρ × ∑ scaled distance)
result.distance     # ∑ scaled distance
result.gradient     # ∇f(B) = ∇g(B∣B)

accuracy = sum( MVDA.classify(problem, X) .== targets ) / length(targets) * 100;
println("Training accuracy is ", accuracy, "%.") # should be around 90-97%
```

## Cross-Validation

The function `cv_MVDA` can be used to tune `ϵ` and `sparsity` via $k$-fold cross-validation.
It supports the same optional arguments as `fit_MVDA`.

```julia
using MVDA, Random, Statistics, Plots

# create the problem instance
df = MVDA.dataset("iris")
targets, X = Vector(df[!,1]), Matrix{Float64}(df[!,2:end])
problem = MVDAProblem(targets, X)
n, p, c = MVDA.probdims(problem)

# IMPORTANT: initialize coefficients
randn!(problem.coeff.all)

ϵ_grid = range(1e-2, 0.5 * sqrt(2*c/(c-1)), length=10)
s_grid = [0.0, 0.25, 0.5, 0.75]

result = cv_MVDA(MMSVD(), problem, ϵ_grid, s_grid,
    nfolds=10,      # number of folds
    at=0.8,         # proportion in training set; rest goes to holdout set for testing
    rtol=0.0,
    gtol=1e-8,
    dtol=1e-6,
);

result.epsilon;     # ϵ_grid
result.sparsity;    # s_grid
result.score;       # 3D array of [Tr, V, T] where Tr, V, and T are training, validation, and testing errors as percentages, respectively.

# average validation error over folds
avg_score = map(x -> x[2], dropdims(mean(result.score, dims=3), dims=3))

# select (sparsity, ϵ) pair minimizing validation error
i,j=Tuple(argmin(avg_score))

# visualize errors over (sparsity, ϵ) pairs as a contour map
contourf(result.epsilon, result.sparsity .* 100, avg_score,
    color=:vik,
    xlabel="ϵ",
    ylabel="Sparsity (%)",
    title="Validation Error",
    yticks=0:10:100,
    clims=(0,100)
)

# highlight the optimal pair
scatter!((result.epsilon[j], result.sparsity[i]*100),
    legend=false,
    marker=:circ,
    markersize=8,
    markercolor=:white,
    grid=false
)
```

## Running package tests

In Julia, with the `MVDA` environment activated, enter package mode and then run the `test` command:

```julia
(MVDA) pkg> test # everything should pass!
```
