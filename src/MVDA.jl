module MVDA

using DataFrames: copy, copyto!
using DataDeps, CSV, DataFrames, CodecZlib
using Parameters, Printf, MLDataUtils, ProgressMeter
using LinearAlgebra, Random, Statistics, StableRNGs

import Base: show, iterate

##### DATA #####

#=
Uses DataDeps to download data as needed.
Inspired by UCIData.jl: https://github.com/JackDunnNZ/UCIData.jl
=#

const DATA_DIR = joinpath(@__DIR__, "data")

"""
`list_datasets()`

List available datasets in MVDA.
"""
list_datasets() = map(x -> splitext(x)[1], readdir(DATA_DIR))

function __init__()
    for dataset in list_datasets()
        include(joinpath(DATA_DIR, dataset * ".jl"))
    end
end

"""
`dataset(str)`

Load a dataset named `str`, if available. Returns data as a `DataFrame` where
the first column contains labels/targets and the remaining columns correspond to
distinct features.
"""
function dataset(str)
    # Locate dataset file.
    dataset_path = @datadep_str str
    file = readdir(dataset_path)
    index = findfirst(x -> occursin("data.", x), file)
    if index isa Int
        dataset_file = joinpath(dataset_path, file[index])
    else # is this unreachable?
        error("Failed to locate a data.* file in $(dataset_path)")
    end
    
    # Read dataset file as a DataFrame.
    df = if splitext(dataset_file)[2] == ".csv"
        CSV.read(dataset_file, DataFrame)
    else # assume .csv.gz
        open(GzipDecompressorStream, dataset_file, "r") do stream
            CSV.read(stream, DataFrame)
        end
    end
    return df
end

"""
Process the dataset located at the given `path`.

This is an extra step to give fine-grain control in generating files with DataDeps.jl.
"""
function process_dataset(path::AbstractString; header=false, missingstrings="", kwargs...)
    input_df = CSV.read(path, DataFrame, header=header, missingstrings=missingstrings)
    process_dataset(input_df; kwargs...)
    rm(path)
end

"""
Final step in processing the given dataset `input_df`.

This standardizes cached files that live in ~/.julia/datadeps so that labels/targets appear in first column
followed by features in the remaining columns.
We also check for uniqueness in features.
"""
function process_dataset(input_df::DataFrame;
    target_index=-1,
    feature_indices=1:0,
    ext=".csv")
    # Build output DataFrame.
    output_df = DataFrame()
    output_df.target = input_df[!, target_index]
    output_df = hcat(output_df, input_df[!, feature_indices], makeunique=true)
    output_cols = [ :target; [Symbol("x", n) for n in eachindex(feature_indices)] ]
    rename!(output_df, output_cols)
    dropmissing!(output_df)
    
    # Write to disk.
    output_path = "data" * ext
    if ext == ".csv"
        CSV.write(output_path, output_df, delim=',', writeheader=true)
    elseif ext == ".csv.gz"
        open(GzipCompressorStream, output_path, "w") do stream
            CSV.write(stream, output_df, delim=",", writeheader=true)
        end
    else
        error("Unknown file extension option '$(ext)'")
    end
end

##### IMPLEMENTATION #####

include("utilities.jl")
include("projections.jl")
include("problem.jl")

abstract type AbstractMMAlg end

__mm_init__(algorithm::AbstractMMAlg, problem, extras) = not_implemented(algorithm, "Initialization step")
__mm_iterate__(algorithm::AbstractMMAlg, problem, ϵ, ρ, k, extras) = not_implemented(algorithm, "Iteration step")
__mm_iterate__(algorithm::AbstractMMAlg, problem, ϵ, λ, extras) = not_implemented(algorithm, "Iteration step (regularized)")
__mm_iterate__(algorithm, problem, ϵ, δ, λ₁, λ₂, extras) = not_implemented(algorithm, "Iteration step (Euclidean)")
__mm_update_sparsity__(algorithm::AbstractMMAlg, problem, ϵ, ρ, k, extras) = not_implemented(algorithm, "Update sparsity step")
__mm_update_rho__(algorithm::AbstractMMAlg, problem, ϵ, ρ, k, extras) = not_implemented(algorithm, "Update ρ step")
__mm_update_lambda__(algorithm::AbstractMMAlg, problem, ϵ, λ, extras) = not_implemented(algorithm, "Update λ step")

include(joinpath("algorithms", "SD.jl"))
include(joinpath("algorithms", "MMSVD.jl"))
include(joinpath("algorithms", "CyclicVDA.jl"))

const DEFAULT_ANNEALING = geometric_progression
const DEFAULT_CALLBACK = __do_nothing_callback__
const DEFAULT_SCORE_FUNCTION = prediction_error

"""
```
fit_MVDA(algorithm, problem, ϵ, s; kwargs...)
```

Solve optimization problem at sparsity levels `s` and `ϵ`-insensitive pseudodistances.
Solution is obtained via a proximal distance `algorithm` that gradually anneals parameter estimates
toward the target sparsity set.
"""
function fit_MVDA(algorithm, problem, ϵ::Real, s::Union{Real,AbstractVector{<:Real}}; kwargs...)
    # Initialize any additional data structures.
    extras = __mm_init__(algorithm, problem, nothing)

    fit_MVDA!(algorithm, problem, ϵ, s, extras, (true,false,); kwargs...)
end

"""
```
fit_MVDA!(algorithm, problem, ϵ, s, [extras], [update_hyperparams]; kwargs...)
```

Same as `fit_MVDA(algorithm, problem, ϵ, s)`, but with preallocated data structures in `extras`.
The caller should specify whether to update data structures depending on `s` (default=`true`).
"""
function fit_MVDA!(algorithm, problem, ϵ::Real, s::Union{Real,AbstractVector{<:Real}}, extras=nothing, update_extras::NTuple{2,Bool}=(true,false,);
    nouter::Int=100,
    dtol::Real=1e-6,
    rtol::Real=1e-6,
    rho_init::Real=1.0,
    rho_max::Real=1e8,
    rhof::Function=DEFAULT_ANNEALING,
    verbose::Bool=false,
    cb::Function=DEFAULT_CALLBACK,
    kwargs...
    )
    # Check for missing data structures.
    if extras isa Nothing
        error("Detected missing data structures for algorithm $(algorithm).")
    end

    # Get problem info and extra data structures.
    @unpack intercept, coeff, coeff_prev, proj = problem
    @unpack apply_projection = extras
    n, p, c = probdims(problem)
    
    # Fix model size(s).
    if s isa Real
        k = [sparsity_to_k(s, p) for _ in 1:c-1]
    else
        k = sparsity_to_k.(s, p)
    end

    # Initialize ρ and iteration count.
    ρ = rho_init
    iters = 0

    # Update data structures due to hyperparameters.
    update_extras[1] && __mm_update_sparsity__(algorithm, problem, ϵ, ρ, k, extras)
    update_extras[2] && __mm_update_rho__(algorithm, problem, ϵ, ρ, k, extras)

    # Check initial values for loss, objective, distance, and norm of gradient.
    copyto!(proj.all, coeff.all)
    apply_projection(proj.all, k, on=:col, intercept=intercept)
    init_result = __evaluate_objective__(problem, ϵ, ρ, extras)
    result = SubproblemResult(0, init_result)
    cb(0, problem, ϵ, ρ, k, result)
    old = result.distance

    for iter in 1:nouter
        # Solve minimization problem for fixed rho.
        verbose && print("\n",iter,"  ρ = ",ρ)
        result = fit_MVDA!(algorithm, problem, ϵ, ρ, s, extras, (false,true,); verbose=verbose, cb=cb, kwargs...)

        # Update total iteration count.
        iters += result.iters

        cb(iter, problem, ϵ, ρ, k, result)

        # Check for convergence to constrained solution.
        dist = result.distance
        if dist < dtol || abs(dist - old) < rtol * (1 + old)
            break
        else
          old = dist
        end
                
        # Update according to annealing schedule.
        ρ = rhof(ρ, iter, rho_max)
    end
    
    # Project solution to the constraint set.
    copyto!(proj.all, coeff.all)
    apply_projection(proj.all, k, on=:col, intercept=intercept)
    loss, obj, dist, gradsq = __evaluate_objective__(problem, ϵ, ρ, extras)

    if verbose
        print("\n\niters = ", iters)
        print("\n∑ᵢ max{0, |yᵢ-Bᵀxᵢ|₂-ϵ}² = ", loss)
        print("\nobjective     = ", obj)
        print("\ndistance      = ", dist)
        println("\n|gradient|² = ", gradsq)
    end

    return SubproblemResult(iters, loss, obj, dist, gradsq)
end

"""
```
fit_MVDA(algorithm, problem, ϵ, ρ, s; kwargs...)
```

Solve the `ρ`-penalized optimization problem at sparsity level `s`.
"""
function fit_MVDA(algorithm, problem, ϵ::Real, ρ::Real, s::Union{Real,AbstractVector{<:Real}}; kwargs...)
    # Initialize any additional data structures.
    extras = __mm_init__(algorithm, problem, nothing)

    fit_MVDA!(algorithm, problem, ϵ, ρ, s, extras, (true,true,); kwargs...)
end

"""
```
fit_MVDA!(algorithm, problem, ϵ, ρ, s, [extras], [update_extras]; kwargs...)
The caller should specify whether to update data structures depending on
    - `lambda` or `s` (default=`true`), and
    - `rho` (default=true).
```

Same as `fit_MVDA!(algorithm, problem, ϵ, ρ, s)`, but with preallocated data structures in `extras`.
"""
function fit_MVDA!(algorithm, problem, ϵ::Real, ρ::Real, s::Union{Real,AbstractVector{<:Real}}, extras=nothing, update_extras::NTuple{2,Bool}=(true,true);
    ninner::Int=10^4,
    gtol::Real=1e-6,
    nesterov_threshold::Int=10,
    verbose::Bool=false,
    cb::Function=DEFAULT_CALLBACK,
    kwargs...
    )
    # Check for missing data structures.
    if extras isa Nothing
        error("Detected missing data structures for algorithm $(algorithm).")
    end

    # Get problem info and extra data structures.
    @unpack intercept, coeff, coeff_prev, proj = problem
    @unpack apply_projection = extras
    n, p, c = probdims(problem)

    # Fix model size(s).
    if s isa Real
        k = [sparsity_to_k(s, p) for _ in 1:c-1]
    else
        k = sparsity_to_k.(s, p)
    end

    # Update data structures due to hyperparameters.
    update_extras[1] && __mm_update_sparsity__(algorithm, problem, ϵ, ρ, k, extras)
    update_extras[2] && __mm_update_rho__(algorithm, problem, ϵ, ρ, k, extras)

    # Check initial values for loss, objective, distance, and norm of gradient.
    copyto!(proj.all, coeff.all)
    apply_projection(proj.all, k, on=:col, intercept=intercept)
    result = __evaluate_objective__(problem, ϵ, ρ, extras)
    cb(0, problem, ϵ, ρ, k, result)
    old = result.objective

    if result.gradient < gtol
        return SubproblemResult(0, result)
    end

    # Initialize iteration counts.
    copyto!(coeff_prev.all, coeff.all)
    iters = 0
    nesterov_iter = 1

    verbose && @printf("\n%-5s\t%-8s\t%-8s\t%-8s\t%-8s", "iter.", "loss", "objective", "distance", "|gradient|²")
    for iter in 1:ninner
        iters += 1

        # Apply the algorithm map to minimize the quadratic surrogate.
        __mm_iterate__(algorithm, problem, ϵ, ρ, k, extras)

        # Update loss, objective, distance, and gradient.
        copyto!(proj.all, coeff.all)
        apply_projection(proj.all, k, on=:col, intercept=intercept)
        result = __evaluate_objective__(problem, ϵ, ρ, extras)

        cb(iter, problem, ϵ, ρ, k, result)

        if verbose
            @printf("\n%4d\t%4.3e\t%4.3e\t%4.3e\t%4.3e", iter, result.loss, result.objective, result.distance, result.gradient)
        end

        # Assess convergence.
        obj = result.objective
        gradsq = result.gradient
        if gradsq < gtol
            break
        elseif iter < ninner
            needs_reset = iter < nesterov_threshold || obj > old
            nesterov_iter = __apply_nesterov__!(coeff.all, coeff_prev.all, nesterov_iter, needs_reset)
            old = obj
        end
    end

    return SubproblemResult(iters, result)
end

"""
```
cv_MVDA(algorithm, problem, ϵ_grid, s_grid; kwargs...)
```

Compute scores for multiple models parameterized by hyperparameters `s` and `ϵ` via K-fold cross-validation.

- `problem` should enter with an initial guess for model parameters (i.e. in `problem.coeff`).
- `ϵ_grid` should be an increasing sequence of nonnegative values.
- `s_grid` should be an increasing sequence of sparsity values (dense to sparse) between 0 and 1.

The default scoring function evaluates the loss, `MSE(y, X*β) + λ * MSE(β,Z*α)`, on the training, validation, and test sets.
"""
function cv_MVDA(algorithm, problem, ϵ_grid, s_grid;
    nfolds::Int=10,
    at::Real=0.8,
    scoref::Function=DEFAULT_SCORE_FUNCTION,
    cb::Function=DEFAULT_CALLBACK,
    kwargs...
    )
    # Split data into cross-validation set and test set.
    @unpack Y, X = problem
    cv_set, test_set = splitobs((Y, X), at=at, obsdim=1)

    # Initialize model object; just used to pass around coefficients.
    n, p, c = probdims(problem)
    T = floattype(problem)
    model = MVDAProblem{T}(Y, X, problem.vertex, problem.label2vertex, problem.vertex2label, problem.intercept,
        deepcopy(problem.coeff), deepcopy(problem.coeff_prev),
        deepcopy(problem.proj), problem.res, deepcopy(problem.grad)
    )

    # Set initial model coefficients.
    init_coeff = problem.coeff.all

    # Initialize the output.
    tmp = scoref(model, test_set, test_set, test_set)
    score = Array{typeof(tmp)}(undef, length(s_grid), length(ϵ_grid), nfolds)

    # Run cross-validation.
    nvals = length(ϵ_grid) * length(s_grid)
    progress_bar = Progress(nfolds*nvals, 1, "Running CV w/ $(algorithm)... ")

    for (k, fold) in enumerate(kfolds(cv_set, k=nfolds, obsdim=1))
        # Retrieve the training set and validation set.
        train_set, validation_set = fold
        train_Y, train_X = train_set
        train_n = size(train_Y, 1)
        train_res = __allocate_res__(T, train_n, p+problem.intercept, c)
        
        # Create a problem object for the training set.
        train_problem = MVDAProblem{T}(copy(train_Y), copy(train_X),
            problem.vertex, problem.label2vertex, problem.vertex2label, problem.intercept,
            deepcopy(problem.coeff), deepcopy(problem.coeff_prev),
            deepcopy(problem.proj), train_res, deepcopy(problem.grad),
        )
        extras = __mm_init__(algorithm, train_problem, nothing)

        for (j, ϵ) in enumerate(ϵ_grid)
            # Set initial model parameters.
            copyto!(train_problem.coeff.all, init_coeff)

            for (i, s) in enumerate(s_grid)
                model_size = [sparsity_to_k(s, p) for _ in 1:c-1]

                # Update data structures due to change in sparsity.
                __mm_update_sparsity__(algorithm, problem, ϵ, one(T), model_size, extras)

                # Obtain solution as function of (s, ϵ).
                result = fit_MVDA!(algorithm, train_problem, ϵ, s, extras, (false, false,); cb=cb, kwargs...)
                copyto!(model.coeff.all, train_problem.coeff.all)
                copyto!(model.proj.all, train_problem.proj.all)

                cb(k, problem, train_problem, (train_set, validation_set, test_set), ϵ, s, model_size, result)

                # Evaluate the solution.
                score[i,j,k] = scoref(model, train_set, validation_set, test_set)

                # Update the progress bar.
                next!(progress_bar, showvalues=[(:fold, k), (:sparsity, s), (:ϵ, ϵ)])
            end
        end
    end

    # Package the result.
    result = (;
        epsilon=ϵ_grid,
        sparsity=s_grid,
        score=score,
    )

    return result
end

function fit_regMVDA(algorithm, problem, ϵ::Real, λ::Real; kwargs...)
    # Initialize any additional data structures.
    extras = __mm_init__(algorithm, problem, nothing)

    fit_regMVDA!(algorithm, problem, ϵ, λ, extras, true; kwargs...)
end

function fit_regMVDA!(algorithm, problem, ϵ, λ, extras=nothing, update_extras::Bool=true;
    ninner::Int=10^4,
    gtol::Real=1e-6,
    nesterov_threshold::Int=10,
    verbose::Bool=false,
    cb::Function=DEFAULT_CALLBACK,
    kwargs...
    )
    # Check for missing data structures.
    if extras isa Nothing
        error("Detected missing data structures for algorithm $(algorithm).")
    end

    # Get problem info and extra data structures.
    @unpack intercept, coeff, coeff_prev, proj = problem
    n, p, c = probdims(problem)

    update_extras && __mm_update_lambda__(algorithm, problem, ϵ, λ, extras)

    # Check initial values for loss, objective, distance, and norm of gradient.
    copyto!(proj.all, coeff.all)
    result = __evaluate_objective_reg__(problem, ϵ, λ, extras)
    # cb(0, problem, ϵ, 0.0, k, result)
    old = result.objective

    if result.gradient < gtol
        return SubproblemResult(0, result)
    end

    # Initialize iteration counts.
    copyto!(coeff_prev.all, coeff.all)
    iters = 0
    nesterov_iter = 1

    verbose && @printf("\n%-5s\t%-8s\t%-8s\t%-8s\t%-8s", "iter.", "loss", "objective", "distance", "|gradient|²")
    for iter in 1:ninner
        iters += 1

        # Apply the algorithm map to minimize the quadratic surrogate.
        __mm_iterate__(algorithm, problem, ϵ, λ, extras)

        # Update loss, objective, distance, and gradient.
        copyto!(proj.all, coeff.all)
        result = __evaluate_objective_reg__(problem, ϵ, λ, extras)

        # cb(iter, problem, ϵ, 0.0, k, result)

        if verbose
            @printf("\n%4d\t%4.3e\t%4.3e\t%4.3e\t%4.3e", iter, result.loss, result.objective, result.distance, result.gradient)
        end

        # Assess convergence.
        obj = result.objective
        gradsq = result.gradient
        if gradsq < gtol
            break
        elseif iter < ninner
            needs_reset = iter < nesterov_threshold || obj > old
            nesterov_iter = __apply_nesterov__!(coeff.all, coeff_prev.all, nesterov_iter, needs_reset)
            old = obj
        end
    end

    return  SubproblemResult(iters, result)
end

function fit_MVDA(algorithm::CyclicVDA, problem, ϵ, δ, λ₁, λ₂;
        niter::Int=10^3,
        atol=1e-4,
    )
    @unpack Y, X, res, coeff = problem
    # δ = 1 / 20
    # ϵ = 1//2 * sqrt(2*c/(c-1))
    n, p, c = probdims(problem)
    μ₁ = n * λ₁
    μ₂ = n * λ₂

    # initialize residuals
    mul!(res.main.all, X, coeff.all)
    axpby!(1.0, Y, -1.0, res.main.all)
    extras = nothing

    full_objective, _, _ = fetch_objective(problem, p+1, 1, ϵ, δ, λ₁, λ₂)
    penalty1 = 0.0
    penalty2 = 0.0
    for j in 1:p # does not include intercept here
        β = view(problem.coeff.all, j, :)
        penalty1 = penalty1 + μ₁ * norm(β, 1)
        penalty2 = penalty2 + μ₂ * norm(β, 2)
    end
    full_objective = full_objective + penalty2 + penalty1

    iters = 0
    for iter in 1:niter
        iters += 1

        __mm_iterate__(algorithm, problem, ϵ, δ, λ₁, λ₂, extras)
        loss, _, _ = fetch_objective(problem, p+1, 1, ϵ, δ, λ₁, λ₂)
        penalty1 = 0.0
        penalty2 = 0.0
        for j in 1:p # does not include intercept here
            β = view(problem.coeff.all, j, :)
            penalty1 = penalty1 + μ₁ * norm(β, 1)
            penalty2 = penalty2 + μ₂ * norm(β, 2)
        end
        objective = loss + penalty2 + penalty1

        if objective > full_objective error("Descent failure") end

        if full_objective - objective < atol break end
        full_objective = objective
    end

    copyto!(problem.proj.all, coeff.all)

    return full_objective, penalty1, penalty2
end

export IterationResult, SubproblemResult
export MVDAProblem, SD, MMSVD, CyclicVDA
export fit_MVDA, fit_MVDA!, cv_MVDA, fit_regMVDA

end
