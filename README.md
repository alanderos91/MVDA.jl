# Multicategory Vertex Discriminant Analysis (MVDA)

A Julia package for sparse classification using a vertex encoding of groups.

## Installation

This package requires Julia 1.7.0 or higher.

```julia
import Pkg
Pkg.add("https://github.com/alanderos91/MVDA.jl")
```

## Demo

Load package and sample data:
```julia
using MVDA

MVDA.list_datasets()        # lists available demo datasets
df = MVDA.dataset("iris")   # loads the `iris` dataset as a DataFrame

# create a problem instance; data has 3 classes + 4 features
L, X = Vector{String}(df[!,1]), Matrix{Float64}(df[!,2:end])
problem = MVDAProblem(L, X;
    kernel=nothing,     # use a linear classifier
    intercept=true,
    encoding=:standard  # encode classes on (1,0,0), (0,1,0), and (0,0,1)
)
```

Fit a sparse VDA model under an $\ell_{0}$-constraint on slope matrix $B$:

```julia
# set loss model
f = PenalizedObjective(SqEpsilonLoss(), SqDistPenalty())

# set hyperparameters
hparams = (;
    epsilon=maximum_deadzone(problem),  # set radius of deadzones around each vertex
    k=3,                                # target number of features
)

# solve with MM algorithm
((iters, result), final_rho) = @time MVDA.solve!(f, MMSVD(), problem, hparams;
    projection_type=HomogeneousL0Projection,    # set constraint handled by distance penalty
    maxiter=10^4,                               # maximum number of inner iterations
    maxrhov=10^2,                               # maximum number of outer iterations
    gtol=1e-3,                                  # converged if |∇f(B)| =< gtol for fixed rho
    rtol=0.0,                                   # converged if abs(dist - old) <= rtol * (1 + old))
    dtol=1e-3,                                  # converged if dist =< dtol
    nesterov=10,                                # minimum number of steps to take WITHOUT Nesterov accel.
    callback=VerboseCallback(5),                # print convergence information every 5 iterations
)
```

Sample output from `VerboseCallback(5)`:

```text
iter 	rho     	risk    	loss    	objective	penalty     |gradient|	distance
   0	1.000e+00	5.457e+02	2.728e+02	2.733e+02	1.000e+00	1.843e+02	1.000e+00
   5	1.000e+00	5.438e-03	2.719e-03	2.779e-03	1.189e-04	2.135e-01	1.090e-02
  10	1.000e+00	1.392e-03	6.959e-04	6.988e-04	5.808e-06	7.311e-02	2.410e-03
  15	1.000e+00	5.317e-04	2.658e-04	2.662e-04	6.706e-07	2.588e-02	8.189e-04
  20	1.000e+00	2.588e-04	1.294e-04	1.296e-04	5.051e-07	8.101e-03	7.107e-04
  25	1.000e+00	1.626e-04	8.129e-05	8.144e-05	3.002e-07	2.641e-03	5.479e-04
  30	1.000e+00	1.123e-04	5.616e-05	5.626e-05	2.049e-07	1.688e-03	4.527e-04
  35	1.000e+00	7.831e-05	3.916e-05	3.923e-05	1.399e-07	1.737e-03	3.741e-04
  40	1.000e+00	5.404e-05	2.702e-05	2.707e-05	9.511e-08	1.070e-03	3.084e-04
```

Access convergence information:

```julia
foreach(println, pairs(result))
# :risk => 4.8321614026236005e-5
# :loss => 2.4160807013118003e-5
# :objective => 2.416135981542583e-5
# :penalty => 9.658005444323104e-12
# :distance => 3.107733168134469e-6
# :gradient => 0.0004930280418838082
```

Check classification accuracy:

```julia
MVDA.accuracy(problem, (L, X)) * 100
# 96.66666666666667
```

Access fitted model parameters:

```julia
problem.coeff_proj.slope
# 4×3 Matrix{Float64}:
#   0.155561   0.111892   0.0922667
#   0.0        0.0        0.0
#  -0.370843  -0.147811  -0.0350137
#   0.209998   0.207149   0.445989

problem.coeff_proj.intercept
# 3-element Vector{Float64}:
#   0.5650525724924076
#   0.19170312700715336
#  -0.6188169252210609
```
