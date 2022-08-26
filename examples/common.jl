using CSV, DataFrames, DelimitedFiles, MLDataUtils, Parameters, ProgressMeter, StableRNGs
using KernelFunctions, LinearAlgebra, MVDA, Statistics, StatsBase
using MKL

using MVDA: ZScoreTransform, NormalizationTransform, NoTransformation,
    maximum_deadzone, make_sparsity_grid, make_log10_grid,
    HistoryCallback

# Assume we are using MKL with 10 Julia threads on a 10-core machine.
#
# MKL and OpenBLAS differ in how they interact with Julia's multithreading.
# By setting number of threads = 1 we have 1 MKL thread per Julia thread (10 total).
# 
# If you remove/comment out the `using MKL` line then you should change this to 10.
#
# See: https://carstenbauer.github.io/ThreadPinning.jl/dev/explanations/blas/#Intel-MKL
# BLAS.set_num_threads(1)

println(
    """
    --- BLAS Configuration ---
    $(BLAS.get_config())

    OPENBLAS_NUM_THREADS=$(get(ENV, "OPENBLAS_NUM_THREADS", 0))
    MKL_NUM_THREADS=$(get(ENV, "MKL_NUM_THREADS", 0))
    BLAS.get_num_threads()=$(BLAS.get_num_threads())

    --- Julia Multi-Threading Configuration ---

    JULIA_NUM_THREADS=$(get(ENV, "JULIA_NUM_THREADS", 0))
    Threads.nthreads()=$(Threads.nthreads())
    """
)

extract_active_subset(problem::MVDAProblem) = extract_active_subset(problem.kernel, problem)

function extract_active_subset(::Nothing, problem::MVDAProblem)
    idx_sample = collect(axes(problem.X, 1))
    idx_feature = MVDA.active_variables(problem)
    _L = view(MVDA.original_labels(problem), idx_sample)
    _X = view(problem.X, idx_sample, idx_feature)
    L, X = getobs((_L, _X), obsdim=1)
    return idx_sample, idx_feature, L, X
end

function extract_active_subset(::Kernel, problem::MVDAProblem)
    idx_sample = MVDA.active_variables(problem)
    idx_feature = collect(axes(problem.X, 2))
    _L = view(MVDA.original_labels(problem), idx_sample)
    _X = view(problem.X, idx_sample, idx_feature)
    L, X = getobs((_L, _X), obsdim=1)
    return idx_sample, idx_feature, L, X
end

function map_original_indices!(dest, src)
    for i in eachindex(dest)
        dest[i] = src[ dest[i] ]
    end
    return dest
end

function get_data_transform(kwargs)
    data_transform = ZScoreTransform # Default
    for (argname, argval) in kwargs
        if argname == :data_transform
            data_transform = argval
        end
    end
    return data_transform
end

function init_callback()
    cb = HistoryCallback()
    MVDA.add_field!(cb, :iters, :risk, :loss, :objective, :distance, :penalty, :gradient)
    return cb
end

function save_reports(dir, problem, convergence_data, train_set, test_set, idx_sample, idx_feature)
    @unpack labels = problem
    delim = '\t'
    labels_row = reshape(labels, 1, length(labels))

    # accuracy
    acc_filename = joinpath(dir, "accuracy.out")
    train_acc, test_acc = MVDA.accuracy(problem, train_set), MVDA.accuracy(problem, test_set)
    open(acc_filename, "w") do io
        writedlm(io, ["train" "test"], delim)
        writedlm(io, [train_acc test_acc], delim)
    end

    # confusion matrix
    mat_filename = joinpath(dir, "confusion_matrix.out")
    train_mat, _ = MVDA.confusion_matrix(problem, train_set)
    test_mat, _ = MVDA.confusion_matrix(problem, test_set)
    open(mat_filename, "w") do io
        writedlm(io, ["subset" "true_label" labels_row], delim)
        for i in eachindex(labels)
            writedlm(io, ["train" labels[i] train_mat[i, :]'], delim)
        end
        for i in eachindex(labels)
            writedlm(io, ["test" labels[i] test_mat[i, :]'], delim)
        end
    end

    # probability matrix
    prob_filename = joinpath(dir, "probability_matrix.out")
    train_prob, _ = MVDA.prediction_probabilities(problem, train_set)
    test_prob, _ = MVDA.prediction_probabilities(problem, test_set)
    open(prob_filename, "w") do io
        writedlm(io, ["subset" "true_label" labels_row], delim)
        for i in eachindex(labels)
            writedlm(io, ["train" labels[i] train_prob[i, :]'], delim)
        end
        for i in eachindex(labels)
            writedlm(io, ["test" labels[i] test_prob[i, :]'], delim)
        end
    end

    # convergence history
    history_filename = joinpath(dir, "history.out")
    conv_vals = collect(values(convergence_data))
    conv_keys = collect(keys(convergence_data))
    CSV.write(history_filename, DataFrame(conv_vals, conv_keys))

    # coefficients
    MVDA.save_model(dir, problem)

    # active samples / variables
    active_samples_filename = joinpath(dir, "active_samples.idx")
    open(active_samples_filename, "w") do io
        writedlm(io, reshape(idx_sample, 1, length(idx_sample)))
    end

    active_features_filename = joinpath(dir, "active_features.idx")
    open(active_features_filename, "w") do io
        writedlm(io, reshape(idx_feature, 1, length(idx_feature)))
    end

    return nothing
end

function run(dir, example, input_data, (ne, ng, nl, ns), sparse2dense::Bool=false;
    at::Real=0.8,
    nfolds::Int=5,
    nreplicates::Int=50,
    intercept::Bool=true,
    kernel::Union{Nothing,Kernel}=nothing,
    kwargs...
    )
    #
    SCORE_METRICS = [:time, :train, :validation, :test]

    # Shuffle data before splitting.
    rng = StableRNG(1903)
    data = shuffleobs(input_data, obsdim=1, rng=rng)

    # Create MVDAProblem instance w/o kernel and construct grids.
    problem = MVDAProblem(data[1], data[2], intercept=intercept, kernel=kernel)
    n, p, c = MVDA.probdims(problem)
    n_train = round(Int, n * at * (nfolds-1)/nfolds)
    n_validate = round(Int, n * at * 1/nfolds)
    n_test = round(Int, n * (1-at))
    nvars = ifelse(problem.kernel isa Nothing, p, n_train)

    @info "Created MVDAProblem for $(example)" samples=n features=p classes=c

    # Create grids for hyperparameters.
    e_grid = if ne > 1
        collect(range(0.0, maximum_deadzone(problem), length=ne))
    else
        [maximum_deadzone(problem)]
    end
    g_grid = if ng > 1
        make_log10_grid(-3, 1, ng)
    else
        [0.0]
    end
    s_grid = make_sparsity_grid(nvars, 4, ns)
    l_grid = if nl > 1
        make_log10_grid(-4, 0, nl)
    else
        [1.0]
    end

    # other settings
    scoref = MVDA.prediction_accuracies
    example_dir = joinpath(dir, example)
    if !ispath(example_dir)
        mkpath(example_dir)
        @info "Created directory for example $(example)" output_dir=example_dir
    end

    # Collect data for cross-validation replicates.
    @info "CV split: $(n_train) Train / $(n_validate) Validate / $(n_test) Test"
    
    MVDA.cv_pipeline(MMSVD(), problem;
        e_grid=e_grid,          # deadzone / epsilon
        g_grid=g_grid,          # gamma / scale parameter
        s_grid=s_grid,          # sparsity
        l_grid=l_grid,          # lambda

        dir=example_dir,
        title=example,
        overwrite=true,

        at=at,                  # propagate CV / Test split
        nreplicates=nreplicates,# number of CV replicates
        nfolds=nfolds,          # propagate number of folds
        rng=rng,                # random number generator for reproducibility

        scoref=scoref,          # scoring function; use prediction accuracy
        by=:validation,         # use accuracy on validation set to select hyperparameters
        minimize=false,         # maximize prediction accuracy

        maxrhov=10^2,           # outer iterations
        maxiter=10^5,           # inner iterations
        gtol=1e-3,              # tolerance on gradient for convergence of inner problem
        dtol=1e-3,              # tolerance on distance for convergence of outer problem
        rtol=0.0,               # use strict distance criteria
        nesterov=10,            # delay on Nesterov acceleration

        show_progress=true,

        kwargs...               # propagate other keywords
    )

    # Compile results for the example.
    summary_filename = joinpath(example_dir, "cv-summary.out")


    # Load data and compute time spent solving for paths.
    cv_result = CSV.read(joinpath(example_dir, "cv-result.out"), DataFrame)

    for df in groupby(cv_result, [:title, :algorithm, :replicate, :fold, :sparsity])
        df[!,:time] .= sum(df[!,:time])
    end

    # compute CV scores
    cv_scores = combine(
        groupby(cv_result, [:title, :algorithm, :replicate, :sparsity]),
        SCORE_METRICS .=> [mean sem]
    )

    cols = map(col -> Symbol(col, :_mean), SCORE_METRICS)
    nt = (;
        sparsity=Float64[],
        time=Float64[],
        time_se=Float64[],
        train=Float64[],
        train_se=Float64[],
        validation=Float64[],
        validation_se=Float64[],
        test=Float64[],
        test_se=Float64[],
    )

    for df in groupby(cv_scores, [:title, :algorithm, :replicate])
        vals = [df[!,col] for col in cols]
        result = NamedTuple(zip(SCORE_METRICS, vals))
        i, (_, sparsity_opt,) = MVDA.search_hyperparameters(s_grid, result,
            by=:validation,
            minimize=false,
            is_average=true
        )

        # record values
        push!(nt.sparsity, sparsity_opt)
        push!(nt.time, df[i, :time_mean])
        push!(nt.time_se, df[i, :time_sem])
        push!(nt.train, df[i, :train_mean])
        push!(nt.train_se, df[i, :train_sem])
        push!(nt.validation, df[i, :validation_mean])
        push!(nt.validation_se, df[i, :validation_sem])
        push!(nt.test, df[i, :test_mean])
        push!(nt.test_se, df[i, :test_sem])
    end
    summary_df = DataFrame(
        replicate=1:nreplicates,
        sparsity=nt.sparsity,
        time=nt.time,
        time_se=nt.time_se,
        train=nt.train,
        train_se=nt.train_se,
        validation=nt.validation,
        validation_se=nt.validation_se,
        test=nt.test,
        test_se=nt.test_se,
    )
    CSV.write(summary_filename, summary_df)
    @info "Saved summary file" file=summary_filename

    # Compare sparse, reduced, and full models.
    progress_bar = Progress(nreplicates * 2 + 1; desc="Fitting final models... ", enabled=true)

    modelA_dir = joinpath(example_dir, "modelA") # sparse
    modelB_dir = joinpath(example_dir, "modelB") # reduced
    modelC_dir = joinpath(example_dir, "modelC") # full

    for dir in (modelA_dir, modelB_dir, modelC_dir)
        if !ispath(dir)
            mkpath(dir)
            @info "Created directory $(dir)"
        end
    end

    # Load hyperparameters.
    epsilon, lambda, gamma = MVDA.load_hyperparameters(joinpath(example_dir, "hyperparameters.out"))
    if problem.kernel isa Nothing
        kernel = nothing
    else
        kernel = ScaledKernel(problem.kernel, gamma)
    end

    # Use train-test split to evaluate final models.
    _train_set, _test_set = splitobs(data, at=at, obsdim=1)
    train_set, test_set = getobs(_train_set, obsdim=1), getobs(_test_set, obsdim=1)
    data_transform = get_data_transform(kwargs)
    F = StatsBase.fit(data_transform, train_set[2], dims=1)
    MVDA.__adjust_transform__(F)
    StatsBase.transform!(F, train_set[2])
    StatsBase.transform!(F, test_set[2])

    L_trainA, X_trainA = train_set[1], train_set[2]
    L_trainC, X_trainC = train_set[1], train_set[2]
    original_sample_idx, original_feature_idx = data[2].indices
    train_sample_idx, train_feature_idx = _train_set[2].indices

    # Fit the full model.
    cb = init_callback()
    problemC = MVDAProblem(L_trainC, X_trainC, problem, kernel)
    MVDA.fit!(MMSVD(), problemC, epsilon, lambda;
        maxiter=10^5,
        gtol=1e-3,
        nesterov=10,
        callback=cb,
    )
    next!(progress_bar, showvalues=[(:replicate, 1), (:model, "full")])

    idx_sampleC, idx_featureC, _, _ = extract_active_subset(problemC)
    
    # C -> Train -> Original
    map_original_indices!(idx_sampleC, train_sample_idx)
    map_original_indices!(idx_sampleC, original_sample_idx)

    # C -> Train -> Original
    map_original_indices!(idx_featureC, train_feature_idx)
    map_original_indices!(idx_featureC, original_feature_idx)
    
    save_reports(modelC_dir, problemC, cb.data, train_set, test_set, idx_sampleC, idx_featureC)

    for (rep, sparsity_opt) in enumerate(nt.sparsity)
        # Fit and test sparse model using selected hyperparameters.
        problemA = MVDAProblem(L_trainA, X_trainA, problem, kernel)
        cb = init_callback()
        if iszero(sparsity_opt) 
            MVDA.fit!(MMSVD(), problemA, epsilon, lambda;
                maxiter=10^5,
                gtol=1e-3,
                nesterov=10,
                callback=cb,
            )
        else
            MVDA.fit!(MMSVD(), problemA, epsilon, lambda, sparsity_opt;
                maxiter=10^5,
                gtol=1e-3,
                dtol=1e-3,
                rtol=0.0,
                maxrhov=10^2,
                nesterov=10,
                callback=cb,
            )
        end
        next!(progress_bar, showvalues=[(:replicate, rep), (:model, "sparse")])

        idx_sampleA, idx_featureA, L_trainB, X_trainB = extract_active_subset(problemA)

        # A -> Train -> Original
        map_original_indices!(idx_sampleA, train_sample_idx)
        map_original_indices!(idx_sampleA, original_sample_idx)
    
        # A -> Train -> Original
        map_original_indices!(idx_featureA, train_feature_idx)
        map_original_indices!(idx_featureA, original_feature_idx)

        dir = joinpath(modelA_dir, string(rep))
        !ispath(dir) && mkpath(dir)
        save_reports(dir,
            problemA, cb.data, train_set, test_set, idx_sampleA, idx_featureA)

        # Fit and test reduced model using only the active variables of the sparse model.
        problemB = MVDAProblem(L_trainB, X_trainB, problem, kernel)
        cb = init_callback()
        MVDA.fit!(MMSVD(), problemB, epsilon, lambda;
            maxiter=10^5,
            gtol=1e-3,
            nesterov=10,
            callback=cb,
        )
        next!(progress_bar, showvalues=[(:replicate, rep), (:model, "reduced")])

        idx_sampleB, idx_featureB, _, _ = extract_active_subset(problemB)
        reduced_train_set = (train_set[1], train_set[2][:, idx_featureA])
        reduced_test_set = (test_set[1], test_set[2][:, idx_featureA])

        # B -> A (Train) -> Original
        map_original_indices!(idx_sampleB, idx_sampleA)
        map_original_indices!(idx_sampleB, original_sample_idx)
    
        # B -> A (Train) -> Original
        map_original_indices!(idx_featureB, idx_featureA)
        map_original_indices!(idx_featureB, train_feature_idx)
        map_original_indices!(idx_featureB, original_feature_idx)

        dir = joinpath(modelB_dir, string(rep))
        !ispath(dir) && mkpath(dir)
        save_reports(dir,
            problemB, cb.data, reduced_train_set, reduced_test_set, idx_sampleB, idx_featureB)
    end
    @info "Saved final model results" sparse=modelA_dir reduced=modelB_dir full=modelC_dir

    return nothing
end
