# https://discourse.julialang.org/t/how-to-read-only-the-last-line-of-a-file-txt/68005/10
function read_last(file)
    open(file) do io
        seekend(io)
        seek(io, position(io) - 2)
        while Char(peek(io)) != '\n'
            seek(io, position(io) - 1)
        end
        Base.read(io, Char)
        Base.read(io, String)
    end
end

get_dataset(problem::MVDAProblem) = (original_labels(problem), problem.X)
split_dataset(problem::MVDAProblem, at) = splitobs(get_dataset(problem), at=at, obsdim=1)

number_of_param_vals(grid::AbstractVector) = length(grid) 
number_of_param_vals(grid::NTuple{N,<:Real}) where N = length(grid)
number_of_param_vals(grids::NTuple) = prod(map(length, grids))

function init_report(filename, header, delim, overwrite::Bool)
    if overwrite || !isfile(filename)
        open(filename, "w") do io
            write(io, join(header, delim), '\n')
        end
        replicate = 1
    else
        line = read_last(filename)
        value = split(line, delim)[3]
        replicate = parse(Int, value) + 1
    end
    return replicate
end

function classification_report(problem::MVDAProblem, data)
    return (;
        score=accuracy(problem, data),
        confusion_matrix=MVDA.confusion_matrix(problem, data),
        probability_matrix=MVDA.prediction_probabilities(problem, data),
    )
end

extract_active_subset(problem::MVDAProblem) = extract_active_subset(problem.kernel, problem)

function extract_active_subset(::Nothing, problem::MVDAProblem)
    idx_sample = collect(axes(problem.X, 1))
    idx_feature = MVDA.active_variables(problem)
    _L = view(MVDA.original_labels(problem), idx_sample)
    _X = view(problem.X, idx_sample, idx_feature)
    L, X = getobs((_L, _X), ObsDim.First())
    return (idx_sample, idx_feature), (L, X)
end

function extract_active_subset(::Kernel, problem::MVDAProblem)
    idx_sample = MVDA.active_variables(problem)
    idx_feature = collect(axes(problem.X, 2))
    _L = view(MVDA.original_labels(problem), idx_sample)
    _X = view(problem.X, idx_sample, idx_feature)
    L, X = getobs((_L, _X), ObsDim.First())
    return (idx_sample, idx_feature), (L, X)
end

get_index_set(::Nothing, (idx_sample, idx_feature)) = idx_feature
get_index_set(::Kernel, (idx_sample, idx_feature)) = idx_sample

function add_structural_zeros!(sparse, reduced, idxs)
    idx = get_index_set(sparse.kernel, idxs)

    sparse.coeff.slope[idx, :] .= reduced.coeff.slope
    sparse.coeff.intercept .= reduced.coeff.intercept

    sparse.coeff_proj.slope[idx, :] .= reduced.coeff.slope
    sparse.coeff_proj.intercept .= reduced.coeff.intercept

    return nothing
end

"""
    cv_path(algorithm, problem, grids; [at], [kwargs...])

Split data in `problem` into cross-validation and a test sets, then run CV over the `grids`.

# Keyword Arguments

- `at`: A value between `0` and `1` indicating the proportion of samples/instances used for
  cross-validation, with remaining samples used for a test set (default=`0.8`).
- `nfolds`: The number of folds to run in cross-validation.
- `scoref`: A function that evaluates a classifier over training, validation, and testing sets 
  (default uses misclassification error).
- `show_progress`: Toggles progress bar.
  
  Additional arguments are propagated to `solve_constrained!` and `solve_unconstrained!`. See also [`MVDA.solve_constrained!`](@ref) and [`MVDA.solve_unconstrained!`](@ref).
"""
function cv_path(
    algorithm::AbstractMMAlg,
    problem::MVDAProblem,
    epsilon::Real,
    lambda::Real,
    s_grid::G,
    data::D=get_dataset(problem);
    projection_type::Type=HomogeneousL0Projection,
    nfolds::Int=5,
    scoref::S=DEFAULT_SCORE_FUNCTION,
    show_progress::Bool=true,
    progress_bar::Progress=Progress(nfolds * length(s_grid); desc="Running CV w/ $(algorithm)... ", enabled=show_progress),
    data_transform::Type{T}=ZScoreTransform,
    rho_init=DEFAULT_RHO_INIT,
    kwargs...,
    ) where {D,G,S,T}
    # Sanity checks.
    if any(x -> x < 0 || x > 1, s_grid)
        error("Values in sparsity grid should be in [0,1].")
    end

    # Initialize the output.
    ns = length(s_grid)
    alloc_score_arrays(a, b) = Array{Float64,2}(undef, a, b)
    result = (;
        train=alloc_score_arrays(ns, nfolds),
        validation=alloc_score_arrays(ns, nfolds),
        time=alloc_score_arrays(ns, nfolds),
    )

    # Run cross-validation.
    for (k, fold) in enumerate(kfolds(data, k=nfolds, obsdim=1))
        # Retrieve the training set and validation set.
        train_set, validation_set = fold
        train_L, train_X = getobs(train_set, obsdim=1)
        val_L, val_X = getobs(validation_set, obsdim=1)
        
        # Standardize ALL data based on the training set.
        # Adjustment of transformation is to detect NaNs, Infs, and zeros in transform parameters that will corrupt data, and handle them gracefully if possible.
        F = StatsBase.fit(data_transform, train_X, dims=1)
        __adjust_transform__(F)
        foreach(Base.Fix1(StatsBase.transform!, F), (train_X, val_X))
        
        # Create a problem object for the training set.
        train_problem = MVDAProblem(train_L, train_X, problem)
        extras = __mm_init__(algorithm, projection_type, train_problem, nothing)
        param_sets = (train_problem.coeff, train_problem.coeff_prev, train_problem.coeff_proj)
        
        # Set initial model parameters.
        for coeff_nt in param_sets
            foreach(Base.Fix2(fill!, 0), coeff_nt)
        end

        # Set initial value for rho.
        rho = rho_init

        for (i, s) in enumerate(s_grid)
            # Fit model.
            if iszero(s)
                timed_result = @timed MVDA.solve!(
                    algorithm, train_problem, epsilon, lambda, extras; kwargs...
                )
            else
                timed_result = @timed MVDA.solve_constrained!(
                    algorithm, train_problem, epsilon, lambda, s, extras;
                    projection_type=projection_type,
                    rho_init=rho, 
                    kwargs...
                )
            end
            
            measured_time = timed_result.time # seconds
            result.time[i,k] = measured_time
            _, new_rho = timed_result.value

            # Check if we can use new rho value in the next step of the solution path.
            rho = ifelse(new_rho > 0, new_rho, rho)

            # Evaluate the solution.
            result.train[i,k] = scoref(train_problem, (train_L, train_X))
            result.validation[i,k] = scoref(train_problem, (val_L, val_X))

            # Update the progress bar.
            spercent = string(round(100*s, digits=4), '%')
            next!(progress_bar, showvalues=[(:fold, k), (:sparsity, spercent)])
        end
    end

    return result
end

function init_cv_tune_progbar(algorithm, problem, nfolds, grids::Tuple{G1,G2,G3}, show_progress) where {G1,G2,G3}
    #
    if problem.kernel isa Nothing
        nvals = number_of_param_vals((grids[1], grids[2]))
    else # isa Kernel
        nvals = number_of_param_vals(grids)
    end
    Progress(nfolds * nvals; desc="Running CV w/ $(algorithm)... ", enabled=show_progress)
end

function cv_tune(algorithm::AbstractMMAlg, problem::MVDAProblem, grids::G,
    data::D=get_dataset(problem);
    nfolds::Int=5,
    scoref::S=DEFAULT_SCORE_FUNCTION,
    show_progress::Bool=true,
    progress_bar::Progress=init_cv_tune_progbar(algorithm, problem, nfolds, grids, show_progress),
    data_transform::Type{T}=ZScoreTransform,
    kwargs...,
    ) where {D,G,S,T}
    #
    e_grid, l_grid, g_grid = grids
    # Sanity checks.
    if any(x -> x < 0 || x > 1, e_grid)
        error("Deadzone values should lie in [0,1].")
    end
    if any(<=(0), l_grid)
        error("Lambda values must be positive.")
    end
    if any(<(0), g_grid)
        error("Gamma values must be nonnegative.")
    end

    # Initialize the output.
    if problem.kernel isa Nothing
        dims = (length(e_grid), length(l_grid), 1, nfolds)
    else # isa Kernel
        dims = (length(e_grid), length(l_grid), length(g_grid), nfolds)
    end
    alloc_score_arrays(dims) = Array{Float64,4}(undef, dims)
    result = (;
        train=alloc_score_arrays(dims),
        validation=alloc_score_arrays(dims),
        test=alloc_score_arrays(dims),
        time=alloc_score_arrays(dims),
    )

    # Run cross-validation.
    for (k, fold) in enumerate(kfolds(data, k=nfolds, obsdim=1))
        # Retrieve the training set and validation set.
        train_set, validation_set = fold
        train_L, train_X = getobs(train_set, obsdim=1)
        val_L, val_X = getobs(validation_set, obsdim=1)
        
        # Standardize ALL data based on the training set.
        # Adjustment of transformation is to detect NaNs, Infs, and zeros in transform parameters that will corrupt data, and handle them gracefully if possible.
        F = StatsBase.fit(data_transform, train_X, dims=1)
        __adjust_transform__(F)
        foreach(Base.Fix1(StatsBase.transform!, F), (train_X, val_X))
        
        fit_args = (algorithm, problem, scoref, kwargs)
        data_subsets = (train_L, train_X), (val_L, val_X)
        mutables = (result, progress_bar)
        __cv_tune_loop__(problem.kernel, fit_args, grids, data_subsets, mutables, k)
    end

    return result
end

function __cv_tune_loop__(::Kernel, fit_args::T1, grids::T2, data_subsets::T3, mutables::T4, k::Integer) where {T1,T2,T3,T4}
    #
    (algorithm, problem, scoref, kwargs) = fit_args
    (e_grid, l_grid, g_grid) = grids
    (train_L, train_X), (val_L, val_X) = data_subsets
    (result, progress_bar) = mutables

    for (l, gamma) in enumerate(g_grid)
        # Create a problem object for the training set.
        new_kernel = ScaledKernel(problem.kernel, gamma)
        train_problem = MVDAProblem(train_L, train_X, problem, new_kernel)
        extras = __mm_init__(algorithm, Nothing, train_problem, nothing)
        param_sets = (train_problem.coeff, train_problem.coeff_prev, train_problem.coeff_proj)

        # Set initial model parameters.
        for coeff_nt in param_sets
            foreach(Base.Fix2(fill!, 0), coeff_nt)
        end

        for (i, epsilon) in enumerate(e_grid), (j, lambda) in enumerate(l_grid)
            # Fit model.
            timed_result = @timed MVDA.solve!(
                algorithm, train_problem, epsilon, lambda, extras; kwargs...
            )
                
            measured_time = timed_result.time # seconds
            result.time[i,j,l,k] = measured_time

            # Evaluate the solution.
            result.train[i,j,l,k] = scoref(train_problem, (train_L, train_X))
            result.validation[i,j,l,k] = scoref(train_problem, (val_L, val_X))

            # Update the progress bar.
            next!(progress_bar, showvalues=[(:fold, k), (:epsilon, epsilon), (:lambda, lambda), (:gamma, gamma)])
        end
    end
end

function __cv_tune_loop__(::Nothing, fit_args::T1, grids::T2, data_subsets::T3, mutables::T4, k::Integer) where {T1,T2,T3,T4}
    #
    (algorithm, problem, scoref, kwargs) = fit_args
    (e_grid, l_grid, _) = grids
    (train_L, train_X), (val_L, val_X) = data_subsets
    (result, progress_bar) = mutables

    # Create a problem object for the training set.
    train_problem = MVDAProblem(train_L, train_X, problem)
    extras = __mm_init__(algorithm, Nothing, train_problem, nothing)
    param_sets = (train_problem.coeff, train_problem.coeff_prev, train_problem.coeff_proj)

    gamma = 0.0
    l = 1

    # Set initial model parameters.
    for coeff_nt in param_sets
        foreach(Base.Fix2(fill!, 0), coeff_nt)
    end

    for (i, epsilon) in enumerate(e_grid), (j, lambda) in enumerate(l_grid)
        # Fit model.
        timed_result = @timed MVDA.solve!(
            algorithm, train_problem, epsilon, lambda, extras; kwargs...
        )
            
        measured_time = timed_result.time # seconds
        result.time[i,j,l,k] = measured_time

        # Evaluate the solution.
        result.train[i,j,l,k] = scoref(train_problem, (train_L, train_X))
        result.validation[i,j,l,k] = scoref(train_problem, (val_L, val_X))

        # Update the progress bar.
        next!(progress_bar, showvalues=[(:fold, k), (:epsilon, epsilon), (:lambda, lambda), (:gamma, gamma)])
    end
end

function fit_tuned_model(algorithm, settings, (epsilon, lambda, gamma, sparsity), (train_set, test_set);
    progress_bar=nothing,
    kwargs...
    )
#
    callback = HistoryCallback()
    add_field!(callback, :iters, :risk, :loss, :objective, :distance, :penalty, :gradient, :rho)
    problem = MVDAProblem(train_set[1], train_set[2], settings)

    if iszero(sparsity)
        timed_result = @timed MVDA.solve!(algorithm, problem, epsilon, lambda;
            callback=callback,
            kwargs...
        )
        if progress_bar isa Progress
            next!(progress_bar, showvalues=[(:model, "reduced")])
        end
        else
        timed_result = @timed MVDA.solve_constrained!(algorithm, problem, epsilon, lambda, sparsity;
            callback=callback,
            kwargs...
        )
        if progress_bar isa Progress
            next!(progress_bar, showvalues=[(:model, "sparse")])
        end
    end
    fit_time = timed_result.time
    fit_result = timed_result.value

    train_result = classification_report(problem, train_set)
    test_result = classification_report(problem, test_set)

    return (;
        train=train_result,
        test=test_result,
        problem=problem,
        epsilon=epsilon,
        lambda=lambda,
        gamma=gamma,
        sparsity=sparsity,
        time=fit_time,
        result=fit_result,
        history=callback.data,
    )
end

function cv(algorithm::AbstractMMAlg, input_problem::MVDAProblem, grids::Tuple{G1,G2,G3,G4};
    data::D=split_dataset(input_problem, 0.8),
    nfolds::Int=5,
    scoref::S=DEFAULT_SCORE_FUNCTION,
    by::Symbol=:validation,
    minimize::Bool=false,
    data_transform::Type{T}=ZScoreTransform,
    kwargs...
) where {D,G1,G2,G3,G4,S,T}
    # Split data into train/test.
    train_data, test_data = data
    train_L, train_X = getobs(train_data, obsdim=1)
    test_L, test_X = getobs(test_data, obsdim=1)
    train_set = (train_L, train_X)
    test_set = (test_L, test_X)

    # Extract grids.
    e_grid, l_grid, g_grid, s_grid = grids

    # Tune epsilon, lambda, and gamma jointly.
    tune_problem = MVDAProblem(train_L, train_X, input_problem)
    tune_grids = (e_grid, l_grid, g_grid)
    tune_result = cv_tune(algorithm, tune_problem, tune_grids;
        scoref=scoref,
        nfolds=nfolds,
        data_transform=data_transform,
        kwargs...
    )
    (_, (tune_score, epsilon, lambda, gamma)) = search_hyperparameters(tune_grids, tune_result,
        by=by,
        minimize=minimize,
    )

    # Create problem object for variable selection step.
    if tune_problem.kernel isa Kernel
        new_kernel = ScaledKernel(tune_problem.kernel, gamma)
        var_select_problem = MVDAProblem(train_L, train_X, tune_problem, new_kernel)
    else
        var_select_problem = MVDAProblem(train_L, train_X, tune_problem, nothing)
        gamma = zero(gamma)
    end

    # Run model selection.
    path_result = cv_path(algorithm, var_select_problem, epsilon, lambda, s_grid;
        scoref=scoref,
        nfolds=nfolds,
        data_transform=data_transform,
        kwargs...
    )
    (_, (path_score, sparsity)) = search_sparsity(s_grid, path_result,
        by=by,
        minimize=minimize,
    )

    # Fit sparse and reduced models.
    settings = var_select_problem

    # Final model using the entire dataset (sparse model).
    params = (epsilon, lambda, gamma, sparsity)
    F = StatsBase.fit(data_transform, train_set[2], dims=1)
    __adjust_transform__(F)
    foreach(Base.Fix1(StatsBase.transform!, F), (train_set[2], test_set[2]))
    fit_result = fit_tuned_model(algorithm, settings, params, (train_set, test_set); kwargs...)

    # Final model using the reduced dataset (reduced model).
    params = (epsilon, lambda, gamma, zero(sparsity))
    (idx_sample, idx_feature), _ = extract_active_subset(fit_result.problem)

    r_train_set = (train_set[1][idx_sample], train_set[2][idx_sample, idx_feature])
    r_test_set = (test_set[1], test_set[2][:, idx_feature])

    tmp = fit_tuned_model(algorithm, settings, params, (r_train_set, r_test_set); kwargs...)
    reduced_problem = MVDAProblem(train_set[1], train_set[2], settings)
    add_structural_zeros!(reduced_problem, tmp.problem, (idx_sample, idx_feature))
    reduced_result = (; tmp..., problem=reduced_problem,)

    return (;
        tune=(; score=tune_score, result=tune_result,),
        path=(; score=path_score, result=path_result,),
        fit=fit_result,
        reduced=reduced_result,
        epsilon=epsilon,
        lambda=lambda,
        gamma=gamma,
        sparsity=sparsity,
    )
end

function repeated_cv(algorithm::AbstractMMAlg, problem::MVDAProblem, grids::Tuple{G1,G2,G3,G4};
    at::Real=0.8,
    nfolds::Int=5,
    nreplicates::Int=10,
    show_progress::Bool=true,
    rng::RNG=StableRNG(1903),
    dir::String=mktempdir(pwd),
    title::String="Example",
    overwrite::Bool=false,
    kwargs...) where {G1,G2,G3,G4,RNG}
#
    nvals = nreplicates * nfolds * number_of_param_vals((grids[1], grids[2], grids[3]))
    nvals += nreplicates * nfolds * number_of_param_vals(grids[4])
    nvals += 2
    progress_bar = Progress(nvals; desc="Repeated CV... ", enabled=show_progress)

    # Split data into randomized cross-validation and test sets.
    unshuffled_cv_set, test_set = split_dataset(problem, at)

    # Replicate CV procedure several times.
    for i in 1:nreplicates
        # Shuffle cross validation set to permute the train/validation sets.
        cv_set = shuffleobs(unshuffled_cv_set, obsdim=1, rng=rng)

        # Run cross validation pipeline.
        result = cv(algorithm, problem, grids;
            data=(cv_set, test_set),
            progress_bar=progress_bar,
            nfolds=nfolds,
            kwargs...
        )

        if i == 1
            save_cv_results(dir, title, algorithm, grids, result; overwrite=overwrite)
        else
            save_cv_results(dir, title, algorithm, grids, result; overwrite=false)
        end
    end

    @info "Saved CV results to disk" title=title dir=dir overwrite=overwrite

    return nothing
end

function search_hyperparameters(grids::Tuple{G1,G2,G3}, result::NamedTuple;
    by::Symbol=:validation,
    minimize::Bool=false,
    is_average::Bool=false,
) where {G1,G2,G3}
    # Extract score data.
    if is_average
        data = getindex(result, by)
    else
        avg_scores = mean(getindex(result, by), dims=4)
        data = dropdims(avg_scores, dims=4)
    end

    # Sanity checks.
    e_grid, l_grid, g_grid = grids
    ne, nl, ng = length(e_grid), length(l_grid), length(g_grid)
    if size(data) != (ne, nl, ng)
        error("Data in NamedTuple is incompatible with ($ne,$nl,$ng) grid.")
    end

    if minimize
        (best_i, best_j, best_l), best_quad = (0, 0, 0,), (Inf, Inf, Inf, Inf)
    else
        (best_i, best_j, best_l), best_quad = (0, 0, 0,), (-Inf, -Inf, -Inf, -Inf)
    end

    epsilons, lambdas, gammas = enumerate(e_grid), enumerate(l_grid), enumerate(g_grid)
    for (l, gamma) in gammas, (j, lambda) in lambdas, (i, epsilon) in epsilons
        quad_score = data[i,j,l]
        proposal = (quad_score, epsilon, lambda, gamma)

        # Check if this is the best pair. Rank by pair_score -> sparsity -> lambda.
        if minimize
            #
            #   quad_score: We want to minimize the CV score; e.g. minimum prediction error.
            #
            t = (quad_score, 1/epsilon, 1/lambda, gamma)
            r = (best_quad[1], 1/best_quad[2], 1/best_quad[3], best_quad[4])
            if t < r
                (best_i, best_j, best_l,), best_quad = (i, j, l,), proposal
            end
        else
            #
            #   quad_score: We want to maximize the CV score; e.g. maximum prediction accuracy.
            #
            t = (quad_score, epsilon, lambda, 1/gamma)
            r = (best_quad[1], best_quad[2], best_quad[3], 1/best_quad[4])
            if t > r
                (best_i, best_j, best_l,), best_quad = (i, j, l,), proposal
            end
        end
    end

    return ((best_i, best_j, best_l,), best_quad)
end

function search_sparsity(grid::AbstractVector, result::NamedTuple;
    by::Symbol=:validation,
    minimize::Bool=false,
    is_average::Bool=false,
)
    # Extract score data.
    if is_average
        data = getindex(result, by)
    else
        avg_scores = mean(getindex(result, by), dims=2)
        data = dropdims(avg_scores, dims=2)
    end

    # Sanity checks.
    if size(data) != size(grid)
        error("Data in NamedTuple is incompatible with ($(length(grid)) × 1) grid.")
    end

    if minimize
        best_i, best_pair = 0, (Inf, Inf)
    else
        best_i, best_pair = 0, (-Inf, -Inf)
    end

    for (i, v) in enumerate(grid)
        score = data[i]
        proposal = (score, v)

        # Check if this is the best value. Rank by score -> hyperparameter value.
        if minimize
            t = (score, 1-v)
            r = (best_pair[1], 1-best_pair[2])
            if t < r
                best_i, best_pair = i, proposal
            end
        else
            t = (score, v)
            r = (best_pair[1], best_pair[2])
            if t > r
                best_i, best_pair = i, proposal
            end
        end
    end

    return (best_i, best_pair)
end

function save_cv_results(dir::String, title::String, algorithm::AbstractMMAlg, grids::G, result::NT;
    overwrite::Bool=false,
) where {G,NT}
#
    if !ispath(dir)
        mkpath(dir)
    end
    # Extract grids.
    e_grid, l_grid, g_grid, s_grid = grids

    # Filenames
    tune_filename = joinpath(dir, "cv_tune.out")
    path_filename = joinpath(dir, "cv_path.out")
    fit_dir = joinpath(dir, "modelA")
    reduced_dir = joinpath(dir, "modelB")

    # Other Setttings/Parameters
    delim = ','
    alg = string(typeof(algorithm))
    epsilon = result.epsilon
    lambda = result.lambda
    gamma = result.gamma

    # CV Tune
    tune_header = ("title", "algorithm", "replicate", "fold", "epsilon", "lambda", "gamma", "time", "train", "validation",)
    replicate = init_report(tune_filename, tune_header, delim, overwrite)
    open(tune_filename, "a") do io
        r = result.tune.result
        is, js, ls, ks = axes(r.time)
        for k in ks, l in ls, j in js, i in is
            cv_data = (title, alg, replicate, k, e_grid[i], l_grid[j], g_grid[l],
                r.time[i,j,l,k],
                r.train[i,j,l,k],
                r.validation[i,j,l,k],
            )
            write(io, join(cv_data, delim), '\n')
        end
        flush(io)
    end

    # CV Path
    path_header = ("title", "algorithm", "replicate", "fold", "epsilon", "lambda", "gamma", "sparsity", "time", "train", "validation",)
    replicate = init_report(path_filename, path_header, delim, overwrite)
    open(path_filename, "a") do io
        r = result.path.result
        is, ks = axes(r.time)
        for k in ks, i in is
            cv_data = (title, alg, replicate, k, epsilon, lambda, gamma, s_grid[i],
                r.time[i,k],
                r.train[i,k],
                r.validation[i,k],
            )
            write(io, join(cv_data, delim), '\n')
        end
        flush(io)
    end

    # Model A: sparse model
    save_fit_results(fit_dir, title, algorithm, result.fit)

    # Model B: reduced model
    save_fit_results(reduced_dir, title, algorithm, result.reduced)

    return nothing
end

function save_fit_results(dir::String, title::String, algorithm::AbstractMMAlg, result::NT;
    overwrite::Bool=false,
) where NT
#
    if !ispath(dir)
        mkpath(dir)
    end

    # Filenames
    fit_filename = joinpath(dir, "summary.out")

    # Other Setttings/Parameters
    delim = ','
    alg = string(typeof(algorithm))
    epsilon = result.epsilon
    lambda = result.lambda
    gamma = result.gamma
    sparsity = result.sparsity
    labels = result.problem.labels

    # Fit Result
    fit_header = ("title", "algorithm", "replicate", "epsilon", "lambda", "gamma", "sparsity", "active_variables", "time", "train", "test",)
    replicate = init_report(fit_filename, fit_header, delim, overwrite)
    open(fit_filename, "a") do io
        fit_data = (title, alg, replicate, epsilon, lambda, gamma, sparsity,
            count_active_variables(result.problem),
            result.time,
            result.train.score,
            result.test.score,
        )
        write(io, join(fit_data, delim), '\n')
    end

    # Additional files nested within directory.
    rep = string(replicate)
    rep_dir = joinpath(dir, rep)
    if !ispath(rep_dir)
        mkpath(rep_dir)
    end
    mat_filename = joinpath(rep_dir, "confusion_matrix.out")
    prob_filename = joinpath(rep_dir, "probability_matrix.out")
    history_filename = joinpath(rep_dir, "history.out")

    # Confusion Matrix
    train_mat, _ = result.train.confusion_matrix
    test_mat, _ = result.test.confusion_matrix
    open(mat_filename, "w") do io
        header = ("subset", "true/predicted", labels...)
        write(io, join(header, delim), '\n')
        for i in eachindex(labels)
            row_data = ("train", labels[i], train_mat[i, :]...)
            write(io, join(row_data, delim), '\n')
        end
        for i in eachindex(labels)
            row_data = ("test", labels[i], test_mat[i, :]...)
            write(io, join(row_data, delim), '\n')
        end
    end

    # Probability Matrix
    train_prob, _ = result.train.probability_matrix
    test_prob, _ = result.test.probability_matrix
    open(prob_filename, "w") do io
        header = ("subset", "true/predicted", labels...)
        write(io, join(header, delim), '\n')
        for i in eachindex(labels)
            row_data = ("train", labels[i], train_prob[i, :]...)
            write(io, join(row_data, delim), '\n')
        end
        for i in eachindex(labels)
            row_data = ("test", labels[i], test_prob[i, :]...)
            write(io, join(row_data, delim), '\n')
        end
    end

    # Convergence History
    open(history_filename, "w") do io
        header = ("iters", "risk", "loss", "objective", "distance", "penalty", "gradient", "rho")
        write(io, join(header, delim), '\n')
        h = NamedTuple(result.history)
        for i in eachindex(h.iters)
            row_data = (h.iters[i], h.risk[i], h.loss[i], h.objective[i],
                h.distance[i], h.penalty[i], h.gradient[i], h.rho[i],
            )
            write(io, join(row_data, delim), '\n')
        end
    end

    # Model Coefficients
    MVDA.save_model(rep_dir, result.problem)

    return nothing
end
