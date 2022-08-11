"""
    cv(algorithm, problem, grids; [at], [kwargs...])

Split data in `problem` into cross-validation and a test sets, then run cross-validation over the `grids`.

# Keyword Arguments

- `at`: A value between `0` and `1` indicating the proportion of samples/instances used for cross-validation, with remaining samples used for a test set (default=`0.8`).

See also: [`MVDA.cv(algorithm::AbstractMMAlg, problem::MVDAProblem, grids::Tuple{E,S}, dataset_split::Tuple{Any,Any})`](@ref)
"""
function cv(algorithm::AbstractMMAlg, problem::MVDAProblem, grids::Tuple{E,S}; at::Real=0.8, kwargs...) where {E,S}
    # Split data into cross-validation and test sets.
    @unpack p, Y, X, intercept = problem
    dataset_split = splitobs((Y, view(X, :, 1:p)), at=at, obsdim=1)
    MVDA.cv(algorithm, problem, grids, dataset_split; kwargs...)
end

"""
    cv(algorithm, problem, grids, dataset_split; [kwargs...])

Run k-fold cross-validation over hyperparameters `(ϵ, s)` for deadzone radius and sparsity level, respectively.

The given `problem` should enter with initial model parameters in `problem.coeff.all`.
Hyperparameters are specified in `grids = (ϵ_grid, s_grid)`, and data subsets are given as `dataset_split = (cv_set, test_set)`.

# Keyword Arguments

- `nfolds`: The number of folds to run in cross-validation.
- `scoref`: A function that evaluates a classifier over training, validation, and testing sets (default uses misclassification error).
- `show_progress`: Toggles progress bar.

Additional arguments are propagated to `fit` and `anneal`. See also [`MVDA.fit`](@ref) and [`MVDA.anneal`](@ref).
"""
function cv(algorithm::AbstractMMAlg, problem::MVDAProblem, grids::Tuple{E,S}, dataset_split::Tuple{Any,Any};
    lambda::Real=1e-3,
    maxiter::Int=10^4,
    tol::Real=1e-4,
    nfolds::Int=5,
    scoref::Function=DEFAULT_SCORE_FUNCTION,
    cb::Function=DEFAULT_CALLBACK,
    show_progress::Bool=true,
    kwargs...) where {E,S}
    # Initialize the output.
    cv_set, test_set = dataset_split
    ϵ_grid, s_grid = grids
    nϵ, ns = length(ϵ_grid), length(s_grid)
    alloc_score_arrays(a, b, c) = [Matrix{Float64}(undef, a, b) for _ in 1:c]
    result = (;
        train=alloc_score_arrays(ns, nϵ, nfolds),
        validation=alloc_score_arrays(ns, nϵ, nfolds),
        test=alloc_score_arrays(ns, nϵ, nfolds),
        time=alloc_score_arrays(ns, nϵ, nfolds),
    )

    # Run cross-validation.
    if show_progress
        progress_bar = Progress(nfolds * nϵ * ns, 1, "Running CV w/ $(algorithm)... ")
    end

    for (k, fold) in enumerate(kfolds(cv_set, k=nfolds, obsdim=1))
        # Retrieve the training set and validation set.
        # TODO: Does this guarantee copies?
        train_set, validation_set = fold
        train_Y, train_X = getobs(train_set, obsdim=1)
        val_Y, val_X = getobs(validation_set, obsdim=1)
        test_Y, test_X = getobs(test_set, obsdim=1)
        
        # Standardize ALL data based on the training set.
        F = StatsBase.fit(ZScoreTransform, train_X, dims=1)
        has_nan = any(isnan, F.scale) || any(isnan, F.mean)
        has_inf = any(isinf, F.scale) || any(isinf, F.mean)
        has_zero = any(iszero, F.scale)
        if has_nan
            error("Detected NaN in z-score.")
        elseif has_inf
            error("Detected Inf in z-score.")
        elseif has_zero
            for idx in eachindex(F.scale)
                x = F.scale[idx]
                F.scale[idx] = ifelse(iszero(x), one(x), x)
            end
        end

        foreach(X -> StatsBase.transform!(F, X), (train_X, val_X, test_X))
        
        # Create a problem object for the training set.
        train_idx, _ = parentindices(train_set[1])
        train_problem = change_data(problem, train_Y, train_X)
        extras = __mm_init__(algorithm, train_problem, nothing)

        for (j, ϵ) in enumerate(ϵ_grid)
            # Set initial model parameters.
            set_initial_coefficients!(train_problem, problem, train_idx)
            
            for (i, s) in enumerate(s_grid)
                # Obtain solution as function of (ϵ, s).
                if s != 0.0
                    result.time[k][i,j] = @elapsed MVDA.fit!(algorithm, train_problem, ϵ, s, extras, (true, false,);
                        cb=cb, kwargs...
                    )
                else# s == 0
                    result.time[k][i,j] = @elapsed MVDA.init!(algorithm, train_problem, ϵ, lambda, extras;
                        maxiter=maxiter, gtol=tol, nesterov_threshold=0,
                    )
                end
                copyto!(train_problem.coeff.all, train_problem.proj.all)

                # Evaluate the solution.
                r = scoref(train_problem, (train_Y, train_X), (val_Y, val_X), (test_Y, test_X))
                for (arr, val) in zip(result, r)
                    arr[k][i,j] = val
                end

                # Update the progress bar.
                if show_progress
                    spercent = string(round(100*s, digits=6), '%')
                    next!(progress_bar, showvalues=[(:fold, k), (:sparsity, spercent), (:ϵ, ϵ)])
                end
            end
        end
    end

    return result
end

function cv_estimation(algorithm::AbstractMMAlg, problem::MVDAProblem, grids::Tuple{E,S}; at::Real=0.8, kwargs...) where {E,S}
    # Split data into cross-validation and test sets.
    @unpack p, Y, X, intercept = problem
    dataset_split = splitobs((Y, view(X, :, 1:p)), at=at, obsdim=1)
    MVDA.cv_estimation(algorithm, problem, grids, dataset_split; kwargs...)
end

function cv_estimation(algorithm::AbstractMMAlg, problem::MVDAProblem, grids::Tuple{E,S}, dataset_split::Tuple{Any,Any};
    nreplicates::Int=10,
    show_progress::Bool=true,
    rng::AbstractRNG=StableRNG(1903),
    kwargs...) where {E,S}
    # Retrieve subsets and create index set into cross-validation set.
    cv_set, test_set = dataset_split

    if show_progress
        progress_bar = Progress(nreplicates, 1, "Running CV w/ $(algorithm)... ")
    end

    # Replicate CV procedure several times.
    replicate = NamedTuple[]
    for r in 1:nreplicates
        # Shuffle cross-validation data.
        cv_shuffled = shuffleobs(cv_set, obsdim=1, rng=rng)

        # Run k-fold cross-validation and store results.
        result = MVDA.cv(algorithm, problem, grids, (cv_shuffled, test_set); show_progress=false, kwargs...)
        push!(replicate, result)

        # Update the progress bar.
        if show_progress
            next!(progress_bar, showvalues=[(:replicate, r),])
        end
    end

    return replicate
end
