# From MLBase.jl; until we can migrate away from deprecated packages.
function unique_inverse(A::AbstractArray)
  out = Array{eltype(A)}(undef, 0)
  out_idx = Array{Vector{Int}}(undef, 0)
  seen = Dict{eltype(A), Int}()
  for (idx, x) in enumerate(A)
      if !in(x, keys(seen))
          seen[x] = length(seen) + 1
          push!(out, x)
          push!(out_idx, Int[])
      end
      push!(out_idx[seen[x]], idx)
  end
  out, out_idx
end

struct StratifiedKfold
  n::Int                         #Total number of observations
  permseqs::Vector{Vector{Int}}  #Vectors of vectors of indexes for each stratum
  k::Int                         #Number of splits
  coeffs::Vector{Float64}        #About how many observations per strata are in a val set
  function StratifiedKfold(rng::AbstractRNG, strata, k)
      2 <= k <= length(strata) || error("The value of k must be in [2, length(strata)].")
      strata_labels, permseqs = unique_inverse(strata)
      map( s -> shuffle!(rng, s), permseqs)
      coeffs = Float64[]
      for (stratum, permseq) in zip(strata_labels, permseqs)
          k <= length(permseq) || error("k is greater than the length of stratum $stratum")
          push!(coeffs, length(permseq) / k)
      end
      new(length(strata), permseqs, k, coeffs)
  end
end

StratifiedKfold(strata, k) = StratifiedKfold(Random.GLOBAL_RNG, strata, k)

length(c::StratifiedKfold) = c.k

function Base.iterate(c::StratifiedKfold, s::Int=1)
  (s > c.k) && return nothing
  r = Int[]
  for (permseq, coeff) in zip(c.permseqs, c.coeffs)
      a, b = round.(Integer, [s-1, s] .* coeff)
      append!(r, view(permseq, a+1:b))
  end
  return setdiff(1:c.n, r), s+1
end

struct StratifiedRandomSub
  idxs::Vector{Vector{Int}}    # total length
  sn::Int                      # length of subset
  sns::Vector{Int}             # num from each stratum for each subset
  k::Int                       # number of subsets
  function StratifiedRandomSub(strata, sn, k)
      n = length(strata)
      strata_labels, idxs = unique_inverse(strata)
      sn >= length(strata_labels) || error("sn is too small for all strata to be represented")
      lengths_ord = sortperm(map(length, idxs))
      sns = zeros(Int, n)
      remaining_n = n   # total in the strata we haven't seen yet
      remaining_sn = sn # total room in the subset that hasn't been "assinged" to a stratum yet
      #loop through strata from smallest to largest, making sure there is at least one idx
      #from each strata in each subset:
      for stratum_num in lengths_ord
          stratum_n = length(idxs[stratum_num])
          remaining_proportion = remaining_sn/remaining_n
          stratum_sn = max(round.(Integer, remaining_proportion*stratum_n), 1)
          remaining_n -= stratum_n
          remaining_sn -= stratum_sn
          sns[stratum_num] = stratum_sn
      end
      #@assert mapreduce(sum, +, sns) == sn
      new(idxs, sn, sns, k)
  end
end

length(c::StratifiedRandomSub) = c.k

function iterate(c::StratifiedRandomSub, s::Int=1)
  (s > c.k) && return nothing
  idxs = Array{Int}(undef, 0)
  sizehint!(idxs, c.sn)
  for (stratum_sn, stratum_idxs) in zip(c.sns, c.idxs)
      append!(idxs, sample(stratum_idxs, stratum_sn, replace=false))
  end
  return (sort!(idxs), s + 1)
end

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
split_dataset(problem::MVDAProblem, at) = splitobs(get_dataset(problem), at=at, obsdim=ObsDim.First())

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

    fill!(sparse.coeff.slope, 0)
    sparse.coeff.slope[idx, :] .= reduced.coeff.slope
    sparse.coeff.intercept .= reduced.coeff.intercept

    fill!(sparse.coeff_proj.slope, 0)
    sparse.coeff_proj.slope[idx, :] .= reduced.coeff.slope
    sparse.coeff_proj.intercept .= reduced.coeff.intercept

    return nothing
end

rescale_coefficients!(::NoTransformation, ::Any) = nothing

function rescale_coefficients!(F::ZScoreTransform, prob)
    if prob.kernel isa Nothing
        prob.coeff.slope ./= F.scale
        prob.coeff_proj.slope ./= F.scale

        if prob.intercept
            prob.coeff.intercept .-= prob.coeff.slope' * F.mean
            prob.coeff_proj.intercept .-= prob.coeff.slope' * F.mean
        end
    end
    return nothing
end

get_penalty_hyperparameter_name(::PenalizedObjective{L,RidgePenalty}, ::Any) where L = :lambda

get_penalty_hyperparameter_name(::PenalizedObjective{L,LassoPenalty}, ::Any) where L = :lambda

get_penalty_hyperparameter_name(::PenalizedObjective{L,SqDistPenalty}, ::Type{L0Projection}) where L = :k
get_penalty_hyperparameter_name(::PenalizedObjective{L,SqDistPenalty}, ::Type{HomogeneousL0Projection}) where L = :k
get_penalty_hyperparameter_name(::PenalizedObjective{L,SqDistPenalty}, ::Type{HeterogeneousL0Projection}) where L = :k

get_penalty_hyperparameter_name(::PenalizedObjective{L,SqDistPenalty}, ::Type{L1BallProjection}) where L = :lambda
get_penalty_hyperparameter_name(::PenalizedObjective{L,SqDistPenalty}, ::Type{HomogeneousL1BallProjection}) where L = :lambda
get_penalty_hyperparameter_name(::PenalizedObjective{L,SqDistPenalty}, ::Type{HeterogeneousL1BallProjection}) where L = :lambda

get_penalty_hyperparameter_name(::PenalizedObjective{L,SqDistPenalty}, ::Type{L2BallProjection}) where L = :lambda
get_penalty_hyperparameter_name(::PenalizedObjective{L,SqDistPenalty}, ::Type{HomogeneousL2BallProjection}) where L = :lambda
get_penalty_hyperparameter_name(::PenalizedObjective{L,SqDistPenalty}, ::Type{HeterogeneousL2BallProjection}) where L = :lambda

# dispatch for penalty-free methods like PGD
function get_penalty_hyperparameter_name(::UnpenalizedObjective{L}, ::Type{P}) where {L,P}
    get_penalty_hyperparameter_name(PenalizedObjective{L,SqDistPenalty}(), P)
end

"""
    cv_path(f, algorithm, problem, hparams, grids, [data]; [kwargs...])

Perform variable selection by running CV over the `grids`, using the pre-tuned hyperparameters in `hparams`.

!!! note
The following text documents arguments specific to `cv_path(...)`. You may set additional keyword arguments that are delegated to `MVDA.solve!` in fitting individual models. See [`solve!`](@ref), [`solve_constrained!`](@ref), or [`solve_unconstrained!`](@ref) for details.
!!!

# Arguments

- `f`: An `AbstractVDAModel` defining the objective.
- `algorithm`: Choice of algorithm to minimize the given objective `f`.
- `problem`: A `MVDAProblem` specifying the VDA settings (linear vs nonlinear, intercept, vertex ecoding). If the positional argument `data` is not specified, then the data wrapped by `problem` is used for CV.
- `hparams`: A `NamedTuple` specifying values for hyperparameters unrelated to variable selection. For example, `(; epsilon=maximum_deadzone(problem))`
  fixes the value for deadzone parameter in VDA to its largest possible value.
- `grids`: A `NamedTuple` containing a grid for either `k` (for L0 type penalties) or `lambda` (for L1 and L2 type penalties). For example, `(; k=[16, 8, 4, 2])` can be used to search 
  over models using 16, 8, 4, and 2 predictors (in that order) when using the `projection_type=HomogeneousL0Projection` option. See also [`make_sparsity_grid`](@ref) and [`make_log10_grid`](@ref).
- `data`: The data to use for variable selection. It should enter as a `Tuple`, `(L, X)`, where `L` is a vector of class labels for each sample in the data matrix `X`, given as rows of the matrix. (default: `data=get_dataset(problem)` where `problem` is the given `MVDAProblem`)

# Keyword Arguments

- `projection_type`: Choice of projection type to implicitly enforce a constraint for variable selection.
- `nfolds`: The number of folds to run in cross-validation.
- `scoref`: A function that evaluates a classifier over training, validation, and testing sets 
  (default uses misclassification error).
- `show_progress`: Toggles the progress bar.
- `data_transform`: A data transformation type from StatsBase applied to data subsets in cross validation. (default: `data_transform=ZScoreTransform`)
- `rho_init`: An initial value for the penalty strength parameter, `rho`. (default: `rho_init=DEFAULT_RHO_INIT`)
- `rng`: A random number generator used to make CV runs reproducible. Use RNGs from `StableRNGs` for complete reproducibility. (default: `Random.default_rng()`)
"""
function cv_path(
    f::AbstractVDAModel,
    algorithm::AbstractMMAlg,
    problem::MVDAProblem,
    hparams,
    grids::G,
    data::D=get_dataset(problem);
    projection_type::Type=HomogeneousL0Projection,
    nfolds::Int=5,
    scoref::S=DEFAULT_SCORE_FUNCTION,
    show_progress::Bool=true,
    progress_bar::Progress=Progress(nfolds; desc="Running CV w/ $(algorithm)... ", enabled=show_progress),
    data_transform::Type{T}=ZScoreTransform,
    rho_init=DEFAULT_RHO_INIT,
    rng::AbstractRNG=Random.default_rng(),
    kwargs...,
    ) where {D,G,S,T}
    # Check model; this is ugly
    if f isa UnpenalizedObjective && !(algorithm isa PGD)
        projection_type = Nothing
    end

    # Initialize the output.
    hparam_name = get_penalty_hyperparameter_name(f, projection_type)
    hparam_grid = getindex(grids, hparam_name)
    hparam_keys = (keys(hparams)..., hparam_name)
    ns = length(hparam_grid)
    alloc_score_arrays(a, b) = Array{Float64,2}(undef, a, b)
    result = (;
        train=alloc_score_arrays(ns, nfolds),
        validation=alloc_score_arrays(ns, nfolds),
        time=alloc_score_arrays(ns, nfolds),
        param=hparam_name,
        grid=hparam_grid,
    )

    if progress_bar.desc == "Running CV w/ $(algorithm)... "
        progress_bar = Progress(nfolds * length(hparam_grid); desc=progress_bar.desc, enabled=progress_bar.enabled)
    end

    # Run cross-validation.
    for (k, fold) in enumerate(kfolds(data, k=nfolds, obsdim=1))
    # for (k, idx) in enumerate(StratifiedKfold(data[1], nfolds))
        # Retrieve the training set and validation set.
        # idx2 = setdiff(1:length(data[1]), idx)
        # fold = ((view(data[1], idx), view(data[2], idx, :)), (view(data[1], idx2), view(data[2], idx2, :)))
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
        extras = __mm_init__(algorithm, (projection_type, rng), train_problem, nothing)
        param_sets = (train_problem.coeff, train_problem.coeff_prev, train_problem.coeff_proj)
        
        # Set initial model parameters.
        for coeff_nt in param_sets
            foreach(Base.Fix2(fill!, 0), coeff_nt)
        end

        # Set initial value for rho.
        rho = rho_init

        for (i, s) in enumerate(hparam_grid)
            # Update the progress bar.
            next!(progress_bar, showvalues=[(:fold, k), (hparam_name, s)])

            # Fit model.
            hparam_vals = (values(hparams)..., s)
            _hparams = NamedTuple{hparam_keys}(hparam_vals)
            timed_result = @timed MVDA.solve!(
                f, algorithm, train_problem, _hparams, extras;
                rng=rng,
                projection_type=projection_type,
                rho_init=rho, 
                kwargs...
            )
            
            measured_time = timed_result.time # seconds
            result.time[i,k] = measured_time
            _, new_rho = timed_result.value

            # Check if we can use new rho value in the next step of the solution path.
            rho = ifelse(new_rho > 0, new_rho, rho)

            # Evaluate the solution.
            result.train[i,k] = scoref(train_problem, (train_L, train_X))
            result.validation[i,k] = scoref(train_problem, (val_L, val_X))
        end
    end

    return result
end

function init_cv_tune_progbar(algorithm, problem, nfolds, grids, show_progress)
    #
    if problem.kernel isa Nothing
        nvals = number_of_param_vals(grids.epsilon)
    else # isa Kernel
        nvals = number_of_param_vals((grids.epsilon, grids.gamma))
    end
    Progress(nfolds * nvals; desc="Running CV w/ $(algorithm)... ", enabled=show_progress)
end

"""
    cv_tune(algorithm, problem, hparams, grids, [data]; [kwargs...])

TODO
"""
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
    (e_grid, g_grid) = (grids.epsilon, grids.gamma)
    # Sanity checks.
    if any(x -> x < 0 || x > 1, e_grid)
        error("Deadzone values should lie in [0,1].")
    end
    if any(<(0), g_grid)
        error("Gamma values must be nonnegative.")
    end

    # Initialize the output.
    if problem.kernel isa Nothing
        dims = (length(e_grid), 1, nfolds)
        g_grid = [zero(eltype(g_grid))]
    else # isa Kernel
        dims = (length(e_grid), length(g_grid), nfolds)
    end
    alloc_score_arrays(dims) = Array{Float64,3}(undef, dims)
    result = (;
        train=alloc_score_arrays(dims),
        validation=alloc_score_arrays(dims),
        time=alloc_score_arrays(dims),
        epsilon=e_grid,
        gamma=g_grid,
    )

    # Run cross-validation.
    for (k, fold) in enumerate(kfolds(data, k=nfolds, obsdim=1))
    # for (k, idx) in enumerate(StratifiedKfold(data[1], nfolds))
        # Retrieve the training set and validation set.
        # idx2 = setdiff(1:length(data[1]), idx)
        # fold = ((view(data[1], idx), view(data[2], idx, :)), (view(data[1], idx2), view(data[2], idx2, :)))
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
    (e_grid, g_grid) = (grids.epsilon, grids.gamma)
    (train_L, train_X), (val_L, val_X) = data_subsets
    (result, progress_bar) = mutables

    for (j, gamma) in enumerate(g_grid)
        # Create a problem object for the training set.
        new_kernel = TransformedKernel(problem.kernel, ScaleTransform(gamma))
        train_problem = MVDAProblem(train_L, train_X, problem, new_kernel)
        extras = __mm_init__(algorithm, (Nothing, nothing), train_problem, nothing)
        param_sets = (train_problem.coeff, train_problem.coeff_prev, train_problem.coeff_proj)

        for (i, epsilon) in enumerate(e_grid)
            # Update the progress bar.
            next!(progress_bar, showvalues=[(:fold, k), (:epsilon, epsilon), (:gamma, gamma)])

            # Set initial model parameters.
            for coeff_nt in param_sets
                foreach(Base.Fix2(fill!, 0), coeff_nt)
            end

            # Fit model.
            hparams = (; epsilon=epsilon,)
            timed_result = @timed MVDA.solve!(
                UnpenalizedObjective(SqEpsilonLoss()),
                algorithm, train_problem, hparams, extras; kwargs...
            )
                
            measured_time = timed_result.time # seconds
            result.time[i,j,k] = measured_time

            # Evaluate the solution.
            result.train[i,j,k] = scoref(train_problem, (train_L, train_X))
            result.validation[i,j,k] = scoref(train_problem, (val_L, val_X))
        end
    end
end

function __cv_tune_loop__(::Nothing, fit_args::T1, grids::T2, data_subsets::T3, mutables::T4, k::Integer) where {T1,T2,T3,T4}
    #
    (algorithm, problem, scoref, kwargs) = fit_args
    e_grid = grids.epsilon
    (train_L, train_X), (val_L, val_X) = data_subsets
    (result, progress_bar) = mutables

    # Create a problem object for the training set.
    train_problem = MVDAProblem(train_L, train_X, problem)
    extras = __mm_init__(algorithm, (Nothing, nothing), train_problem, nothing)
    param_sets = (train_problem.coeff, train_problem.coeff_prev, train_problem.coeff_proj)

    gamma = 0.0
    j = 1

    for (i, epsilon) in enumerate(e_grid)
        # Update the progress bar.
        next!(progress_bar, showvalues=[(:fold, k), (:epsilon, epsilon), (:gamma, gamma)])

        # Set initial model parameters.
        for coeff_nt in param_sets
            foreach(Base.Fix2(fill!, 0), coeff_nt)
        end

        # Fit model.
        hparams = (; epsilon=epsilon,)
        timed_result = @timed MVDA.solve!(
            UnpenalizedObjective(SqEpsilonLoss()),
            algorithm, train_problem, hparams, extras; kwargs...
        )
            
        measured_time = timed_result.time # seconds
        result.time[i,j,k] = measured_time

        # Evaluate the solution.
        result.train[i,j,k] = scoref(train_problem, (train_L, train_X))
        result.validation[i,j,k] = scoref(train_problem, (val_L, val_X))
    end
end

function fit_tuned_model(f, algorithm, settings, hparams, (train_set, test_set);
    progress_bar::Union{Nothing,Progress}=nothing,
    kwargs...
)
#
    callback = HistoryCallback()
    add_field!(callback, :iters, :risk, :loss, :objective, :distance, :penalty, :gradient, :rho)
    problem = MVDAProblem(train_set[1], train_set[2], settings)

    param_sets = (problem.coeff, problem.coeff_prev, problem.coeff_proj)
    for coeff_nt in param_sets
        foreach(Base.Fix2(fill!, 0), coeff_nt)
    end

    timed_result = @timed MVDA.solve!(f, algorithm, problem, hparams;
        callback=callback,
        kwargs...
    )
    model_type = f isa UnpenalizedObjective ? "reduced" : "sparse"
    progress_bar isa Progress && next!(progress_bar, showvalues=[(:model, model_type)])

    fit_time = timed_result.time
    fit_result = timed_result.value

    train_result = classification_report(problem, train_set)
    test_result = classification_report(problem, test_set)

    return (;
        train=train_result,
        test=test_result,
        problem=problem,
        hyperparameters=hparams,
        time=fit_time,
        result=fit_result,
        history=callback.data,
    )
end

"""
    cv(f, algorithm, problem, grids; [kwargs]...)

TODO
"""
function cv(
    f::Union{UnpenalizedObjective{LOSS},PenalizedObjective{LOSS,PEN}},
    algorithm::AbstractMMAlg,
    input_problem::MVDAProblem,
    grids::G;
    # keyword argument
    data::D=split_dataset(input_problem, 0.8),
    nfolds::Int=5,
    scoref::S=DEFAULT_SCORE_FUNCTION,
    by::Symbol=:validation,
    minimize::Bool=false,
    data_transform::Type=ZScoreTransform,
    projection_type::Type=HomogeneousL0Projection,
    kwargs...
) where {LOSS,PEN,D,G,S}
    if f isa UnpenalizedObjective && !(algorithm isa PGD)
        projection_type = Nothing
    end

    # Split data into train/test.
    train_data, test_data = data
    train_L, train_X = getobs(train_data, obsdim=1)
    test_L, test_X = getobs(test_data, obsdim=1)
    train_set = (train_L, train_X)
    test_set = (test_L, test_X)

    # Tune epsilon and gamma jointly.
    tune_problem = MVDAProblem(train_L, train_X, input_problem)
    tune_result = cv_tune(algorithm, tune_problem, grids;
        scoref=scoref,
        nfolds=nfolds,
        data_transform=data_transform,
        kwargs...
    )
    (_, (tune_score, epsilon, gamma)) = search_hyperparameters(tune_result,
        by=by,
        minimize=minimize,
        is_average=false
    )

    # Create problem object for variable selection step.
    if tune_problem.kernel isa Kernel
        new_kernel = TransformedKernel(tune_problem.kernel, ScaleTransform(gamma))
        var_select_problem = MVDAProblem(train_L, train_X, tune_problem, new_kernel)
    else
        var_select_problem = MVDAProblem(train_L, train_X, tune_problem, nothing)
        gamma = zero(gamma)
    end
    hparams_path = (
        epsilon=epsilon,
        gamma=gamma,
    )

    # Run model selection.
    path_result = cv_path(f, algorithm, var_select_problem, hparams_path, grids;
        scoref=scoref,
        nfolds=nfolds,
        data_transform=data_transform,
        projection_type=projection_type,
        kwargs...
    )
    hparam_name = get_penalty_hyperparameter_name(f, projection_type)
    hparam_grid = getindex(grids, hparam_name)

    if hparam_name == :k
        (_, (path_score, optimal_parameter)) = search_sparsity(hparam_grid, path_result,
            by=by,
            minimize=minimize,
            is_average=false
        )
    elseif hparam_name == :lambda
        (_, (path_score, optimal_parameter)) = search_lambda(hparam_grid, path_result,
            by=by,
            minimize=minimize,
            is_average=false
        )
    end
    hparam_keys = (:epsilon, :gamma, hparam_name)
    hparam_vals = (epsilon, gamma, optimal_parameter)
    _hparams = NamedTuple{hparam_keys}(hparam_vals)
    hparams=(;
        epsilon=epsilon,
        gamma=gamma,
        lambda=get(_hparams, :lambda, 0.0),
        k=get(_hparams, :k, size(train_set[2], 2)),
    )

    # Fit sparse and reduced models.
    settings = var_select_problem

    # Final model using the entire dataset (sparse model).
    F = StatsBase.fit(data_transform, train_set[2], dims=1)
    __adjust_transform__(F)
    foreach(Base.Fix1(StatsBase.transform!, F), (train_set[2], test_set[2]))
    fit_result = fit_tuned_model(f, algorithm, settings, hparams, (train_set, test_set); projection_type=projection_type, kwargs...)

    # Final model using the reduced dataset (reduced model).
    (idx_sample, idx_feature), _ = extract_active_subset(fit_result.problem)
    r_train_set = (train_set[1][idx_sample], train_set[2][idx_sample, idx_feature])
    r_test_set = (test_set[1], test_set[2][:, idx_feature])
    reduced_model = UnpenalizedObjective(LOSS())
    tmp = fit_tuned_model(reduced_model, algorithm, settings, hparams, (r_train_set, r_test_set); kwargs...)
    reduced_problem = MVDAProblem(train_set[1], train_set[2], settings)
    add_structural_zeros!(reduced_problem, tmp.problem, (idx_sample, idx_feature))
    reduced_result = (; tmp..., problem=reduced_problem,)

    # Rescale the coefficients
    rescale_coefficients!(F, fit_result.problem)
    rescale_coefficients!(F, reduced_result.problem)

    return (;
        model=f,
        projection=projection_type,
        kernel=settings.kernel,
        tune=(; score=tune_score, result=tune_result,),
        path=(; score=path_score, result=path_result,),
        fit=fit_result,
        reduced=reduced_result,
        hyperparameters=hparams,
    )
end

"""
    repeated_cv(f, algorithm, problem, grids; [kwargs]...)

TODO
"""
function repeated_cv(f::AbstractVDAModel, algorithm::AbstractMMAlg, problem::MVDAProblem, grids::G;
    at::Real=0.8,
    nfolds::Int=5,
    nreplicates::Int=10,
    show_progress::Bool=true,
    rng::RNG=Random.default_rng(),
    projection_type::Type=HomogeneousL0Projection,
    dir::String=mktempdir(pwd()),
    title::String="Example",
    overwrite::Bool=false,
    kwargs...
    ) where {G,RNG}
#
    e_grid, g_grid = grids.epsilon, grids.gamma
    hparam_name = get_penalty_hyperparameter_name(f, projection_type)
    hparam_grid = getindex(grids, hparam_name)

    if problem.kernel isa Nothing
        dims = (length(e_grid), 1, nfolds)
        g_grid = [zero(eltype(g_grid))]
    else # isa Kernel
        dims = (length(e_grid), length(g_grid), nfolds)
    end

    nvals_tune = number_of_param_vals((e_grid, g_grid))
    nvals_path = number_of_param_vals(hparam_grid)
    nvals_per_rep = nfolds * (nvals_tune + nvals_path) + 2
    nvals = nreplicates * nvals_per_rep
    progress_bar = Progress(nvals; desc="Repeated CV", enabled=show_progress)

    unshuffled_set = get_dataset(problem)

    # Replicate CV procedure several times.
    for i in 1:nreplicates
        progress_bar.desc = "Repeated CV | Replicate $(i) / $(nreplicates)"
        # Split data into random training and test sets.
        data = shuffleobs(unshuffled_set, obsdim=1, rng=rng)
        cv_set, test_set = splitobs(data, at=at, obsdim=ObsDim.First())

        # Run cross validation pipeline.
        result = cv(f, algorithm, problem, grids;
            data=(cv_set, test_set),
            progress_bar=progress_bar,
            nfolds=nfolds,
            rng=rng,
            projection_type=projection_type,
            kwargs...
        )

        if i == 1
            save_cv_results(dir, title, algorithm, result; overwrite=overwrite)
        else
            save_cv_results(dir, title, algorithm, result; overwrite=false)
        end
    end
    finish!(progress_bar)

    @info "Saved CV results to disk" title=title dir=dir overwrite=overwrite

    return nothing
end

function search_hyperparameters(result::NamedTuple;
    by::Symbol=:validation,
    minimize::Bool=false,
    is_average::Bool=false,
) where G
    # Extract score data.
    if is_average
        data = getindex(result, by)
    else
        avg_scores = mean(getindex(result, by), dims=3)
        data = dropdims(avg_scores, dims=3)
    end

    if minimize
        (best_i, best_j), best_triple = (0, 0,), (Inf, Inf, Inf)
    else
        (best_i, best_j), best_triple = (0, 0,), (-Inf, -Inf, -Inf)
    end

    for j in axes(data, 2), i in axes(data, 1)
        epsilon, gamma = result.epsilon[i], result.gamma[j]
        triple_score = data[i,j]
        proposal = (triple_score, epsilon, gamma)

        # Check if this is the best triple.
        if minimize
            #
            #   triple_score: We want to minimize the CV score; e.g. minimum prediction error.
            #
            t = (triple_score, 1/epsilon, gamma)
            r = (best_triple[1], 1/best_triple[2], best_triple[3])
            if t < r
                (best_i, best_j,), best_triple = (i, j,), proposal
            end
        else
            #
            #   triple_score: We want to maximize the CV score; e.g. maximum prediction accuracy.
            #
            t = (triple_score, epsilon, 1/gamma)
            r = (best_triple[1], best_triple[2], 1/best_triple[3])
            if t > r
                (best_i, best_j,), best_triple = (i, j,), proposal
            end
        end
    end

    return ((best_i, best_j,), best_triple)
end

function search_sparsity(grid::AbstractVector{Int}, result::NamedTuple;
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
        best_i, best_pair = 0, (Inf, typemax(Int))
    else
        best_i, best_pair = 0, (-Inf, typemin(Int))
    end

    for (i, k) in enumerate(grid)
        score = data[i]
        proposal = (score, k)

        # Check if this is the best value. Rank by score -> hyperparameter value.
        if minimize
            t = (score, 1.0*k)
            r = (best_pair[1], 1.0*best_pair[2])
            if t < r
                best_i, best_pair = i, proposal
            end
        else
            t = (score, 1.0/k)
            r = (best_pair[1], 1.0/best_pair[2])
            if t > r
                best_i, best_pair = i, proposal
            end
        end
    end

    return (best_i, best_pair)
end

function search_lambda(grid::AbstractVector, result::NamedTuple;
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

    for (i, lambda) in enumerate(grid)
        score = data[i]
        proposal = (score, lambda)

        # Check if this is the best value. Rank by score -> hyperparameter value.
        if minimize
            t = (score, 1.0*lambda)
            r = (best_pair[1], 1.0*best_pair[2])
            if t < r
                best_i, best_pair = i, proposal
            end
        else
            t = (score, 1.0/lambda)
            r = (best_pair[1], 1.0/best_pair[2])
            if t > r
                best_i, best_pair = i, proposal
            end
        end
    end

    return (best_i, best_pair)
end

function save_cv_results(dir::String, title::String, algorithm::AbstractMMAlg, result::NT;
    overwrite::Bool=false,
) where NT
#
    if !ispath(dir)
        mkpath(dir)
    end

    # Filenames
    tune_filename = joinpath(dir, "cv_tune.out")
    path_filename = joinpath(dir, "cv_path.out")
    fit_dir = joinpath(dir, "modelA")
    reduced_dir = joinpath(dir, "modelB")

    # Other Setttings/Parameters
    delim = ','
    alg = string(typeof(algorithm))
    hyperparameters = result.hyperparameters

    # CV Tune
    tune_header = ("title", "algorithm", "replicate", "fold", "epsilon", "gamma", "time", "train", "validation",)
    replicate = init_report(tune_filename, tune_header, delim, overwrite)
    open(tune_filename, "a") do io
        r = result.tune.result
        is, js, ks = axes(r.time)
        for k in ks, j in js, i in is
            cv_data = (title, alg, replicate, k, r.epsilon[i], r.gamma[j],
                r.time[i,j,k],
                r.train[i,j,k],
                r.validation[i,j,k],
            )
            write(io, join(cv_data, delim), '\n')
        end
        flush(io)
    end

    # CV Path
    path_header = ("title", "algorithm", "replicate", "fold", "epsilon", "gamma", "lambda", "k", "time", "train", "validation",)
    replicate = init_report(path_filename, path_header, delim, overwrite)
    open(path_filename, "a") do io
        r = result.path.result
        is, ks = axes(r.time)
        for k in ks, i in is
            if r.param == :lambda
                lambda = r.grid[i]
                param_k = hyperparameters.k
            elseif r.param == :k
                lambda = hyperparameters.lambda
                param_k = r.grid[i]
            end
            cv_data = (title, alg, replicate, k, hyperparameters.epsilon, hyperparameters.gamma, lambda, param_k,
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
    hyperparameters = result.hyperparameters
    epsilon = hyperparameters.epsilon
    lambda = hyperparameters.lambda
    gamma = hyperparameters.gamma
    k = hyperparameters.k
    labels = result.problem.labels

    # Fit Result
    fit_header = ("title", "algorithm", "replicate", "epsilon", "lambda", "gamma", "k", "active_variables", "time", "train", "test",)
    replicate = init_report(fit_filename, fit_header, delim, overwrite)
    open(fit_filename, "a") do io
        fit_data = (title, alg, replicate, epsilon, lambda, gamma, k,
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
