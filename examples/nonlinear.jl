using CSV, DataFrames, KernelFunctions, MVDA, Plots, StableRNGs
using LinearAlgebra

BLAS.set_num_threads(8)

run = function(dir, example, data, sparse2dense::Bool=false; at::Real=0.8, kwargs...)
    # Create MVDAProblem instance w/ RBFKernel and construct grids.
    @info "Creating MVDAProblem for $(example)"
    problem = MVDAProblem(data..., intercept=true, kernel=RBFKernel())
    n, p, c = MVDA.probdims(problem)
    n_cv = round(Int, n*at)
    fill!(problem.coeff.all, 1/(n_cv+1))
    ϵ_grid = [MVDA.maximal_deadzone(problem)]
    s_grid = sort!([1-k/n_cv for k in n_cv:-1:0], rev=sparse2dense)
    grids = (ϵ_grid, s_grid)

    # Collect data for cross-validation replicates.
    @info "Cross-Validation"
    rng = StableRNG(1903)
    replicates = MVDA.cv_estimation(MMSVD(), problem, grids;
        at=at,                  # propagate CV / Test split
        rng=rng,                # random number generator for reproducibility
        nouter=10^2,            # outer iterations
        ninner=10^6,            # inner iterations
        gtol=1e-6,              # tolerance on gradient for convergence of inner problem
        dtol=1e-6,              # tolerance on distance for convergence of outer problem
        rtol=0.0,               # use strict distance criteria
        nesterov_threshold=100, # delay on Nesterov acceleration
        show_progress=true,     # display progress over replicates
        kwargs...               # propagate other keywords
    )

    # Write results to disk.
    @info "Writing to disk"
    traversal = sparse2dense ? "S2D" : "D2S"
    partial_filename = joinpath(dir, "$(example)-NL-path=$(traversal)")
    title = "$(example) / $(n) samples / $(p) features / $(c) classes"
    MVDA.save_cv_results(partial_filename*".dat", replicates, grids)

    # Default plot options.
    w, h = default(:size)
    options = (; left_margin=5Plots.mm, size=(1.5*w, 1.5*h),)

    # Summarize CV results over folds + make plot.
    @info "Summarizing over folds"
    cv_results = CSV.read(partial_filename*".dat", DataFrame, header=true)
    cv_paths = MVDA.cv_error(cv_results)
    for metric in (:train, :validation, :test)
        fig = MVDA.plot_cv_paths(cv_paths, metric)
        plot!(fig; title=title, options...)
        savefig(fig, partial_filename*"-replicates=$(metric).png")
    end

    # Construct credible intervals for detailed summary plot.
    @info "Constructing credible intervals"
    cv_intervals = MVDA.credible_intervals(cv_paths)
    for metric in (:train, :validation, :test)
        fig = MVDA.plot_credible_intervals(cv_intervals, metric)
        plot!(fig; title=title, options...)
        savefig(fig, partial_filename*"-summary=$(metric).png")
    end

    return nothing
end

# Nested Circles
n_cv, n_test = 250, 10^3
nsamples = n_cv + n_test
nclasses = 3
data = MVDA.generate_nested_circle(nsamples, nclasses; p=8//10, rng=StableRNG(1903))
run("/home/alanderos/Desktop/VDA/", "circles", data, false;
    at=n_cv/nsamples,   # CV set / Test set split
    nfolds=5,           # number of folds
    nreplicates=100,    # number of CV replicates
)

# Waveform
n_cv, n_test = 375, 10^3
nsamples = n_cv + n_test
nfeatures = 21
data = MVDA.generate_waveform(nsamples, nfeatures; rng=MersenneTwister(1903))
run("/home/alanderos/Desktop/VDA/", "waveform", data, false;
    at=n_cv/nsamples,   # CV set / Test set split
    nfolds=5,           # number of folds
    nreplicates=100,    # number of CV replicates
)

# Zoo
df = MVDA.dataset("zoo")
data = (Vector(df[!,1]), Matrix{Float64}(df[!,2:end]))
run("/home/alanderos/Desktop/VDA/", "zoo", data, false;
    at=0.8,             # CV set / Test set split
    nfolds=5,           # number of folds
    nreplicates=100,    # number of CV replicates
)

# Vowel
df = MVDA.dataset("vowel")
data = (Vector(df[!,1]), Matrix{Float64}(df[!,2:end]))
run("/home/alanderos/Desktop/VDA/", "vowel", data, false;
    at=0.533333,        # CV set / Test set split
    nfolds=5,           # number of folds
    nreplicates=100,    # number of CV replicates
)
