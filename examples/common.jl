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

function get_data_transform(kwargs)
    data_transform = ZScoreTransform # Default
    for (argname, argval) in kwargs
        if argname == :data_transform
            data_transform = argval
        end
    end
    return data_transform
end

function run(dir, example, input_data, (ne, ng, nl, ns), projection_type, preshuffle::Bool=false;
    at::Real=0.8,
    nfolds::Int=5,
    nreplicates::Int=50,
    intercept::Bool=true,
    kernel::Union{Nothing,Kernel}=nothing,
    kwargs...
    )
    # Shuffle data before splitting.
    rng = StableRNG(1903)
    if preshuffle
        data = getobs(shuffleobs(input_data, ObsDim.First(), rng), ObsDim.First())
    else
        data = getobs(input_data, ObsDim.First())
    end

    # Create MVDAProblem instance w/o kernel and construct grids.
    problem = MVDAProblem(data[1], data[2], intercept=intercept, kernel=kernel, encoding=:standard)
    n, p, c = MVDA.probdims(problem)
    n_train = round(Int, n * at * (nfolds-1)/nfolds)
    n_validate = round(Int, n * at * 1/nfolds)
    n_test = round(Int, n * (1-at))
    nvars = ifelse(problem.kernel isa Nothing, p, n_train)

    @info "Created MVDAProblem for $(example)" samples=n features=p classes=c projection=projection_type

    # Epsilon / Deadzone grid.
    e_grid = if ne > 1
        if c > 2
            1e1 .^ range(-3, log10(maximum_deadzone(problem)), length=ne)
        else
            1e1 .^ range(-6, 0, length=ne)
        end
    else
        if c > 2
            [maximum_deadzone(problem)]
        else
            [1e-6]
        end
    end

    # Lambda grid.
    l_grid = if nl > 1
        sort!(make_log10_grid(-6, 6, nl), rev=true) # large values (less shrinkage) to small values (more shrinkage)
    else
        [1.0]
    end

    # Gamma / Scale grid.
    g_grid = if ng > 1
        make_log10_grid(-1, 1, ng)
    else
        [0.0]
    end
    
    # Sparsity grid.
    k_grid = round.(Int, nvars .* (1 .- make_sparsity_grid(nvars, ns)))
    
    grids = (
        epsilon=e_grid,
        lambda=l_grid,
        gamma=g_grid, 
        k=k_grid,
    )

    # other settings
    scoref = MVDA.accuracy
    proj = string(projection_type)
    example_dir = joinpath(dir, example, proj)
    if !ispath(example_dir)
        mkpath(example_dir)
        @info "Created directory for example $(example)" output_dir=example_dir
    end

    # Collect data for cross-validation replicates.
    @info "CV split: $(n_train) Train / $(n_validate) Validate / $(n_test) Test"
    
    model = PenalizedObjective(SqEpsilonLoss(), SqDistPenalty())
    MVDA.repeated_cv(model, MMSVD(), problem, grids;
        dir=example_dir,        # directory to store all results
        title=example,
        overwrite=true,

        at=at,                  # propagate CV / Test split
        nreplicates=nreplicates,# number of CV replicates
        nfolds=nfolds,          # propagate number of folds
        rng=rng,                # random number generator for reproducibility
        
        projection_type=projection_type,

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

    return nothing
end
