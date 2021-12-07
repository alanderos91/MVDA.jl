using CSV, DataFrames, MLDataUtils, KernelFunctions, MVDA, Plots, StableRNGs
using LinearAlgebra, Statistics

BLAS.set_num_threads(8)

add_model_size_guide = function(fig, N)
    # Compute ticks based on maximum model size N.
    if N > 16
        xticks = collect(round.(Int, N .* range(0, 1, length=11)))
    else
        xticks = collect(0:N)
    end
    sort!(xticks, rev=true)

    # Register figure inside main subplot and append extra x-axis.
    model_size_guide = plot(yticks=nothing, xticks=xticks, xlim=(0,N), xlabel="Model Size", xflip=true)
    full_figure = plot(fig, model_size_guide, layout=@layout [a{1.0h}; b{1e-8h}])

    return full_figure
end

run = function(dir, example, data, sparse2dense::Bool=false; at::Real=0.8, nfolds::Int=5, kwargs...)
    # Create MVDAProblem instance w/o kernel and construct grids.
    @info "Creating MVDAProblem for $(example)"
    problem = MVDAProblem(data..., intercept=true, kernel=nothing)
    n, p, c = MVDA.probdims(problem)
    n_train = round(Int, n * at * (nfolds-1)/nfolds)
    n_validate = round(Int, n * at * 1/nfolds)
    n_test = round(Int, n * (1-at))
    fill!(problem.coeff.all, 1/(p+1))
    ϵ_grid = [MVDA.maximal_deadzone(problem)]
    s_grid = sort!([1-k/p for k in p:-1:0], rev=sparse2dense)
    grids = (ϵ_grid, s_grid)

    @info "CV split: $(n_train) Train / $(n_validate) Validate / $(n_test) Test"
    data_subsets = (n_train, n_validate, n_test)
    titles = ["$(example) / $(_n) samples / $(p) features / $(c) classes" for _n in data_subsets]
    metrics = (:train, :validation, :test)

    # Collect data for cross-validation replicates.
    rng = StableRNG(1903)
    replicates = MVDA.cv_estimation(MMSVD(), problem, grids;
        at=at,                  # propagate CV / Test split
        nfolds=nfolds,          # propagate number of folds
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
    partial_filename = joinpath(dir, "$(example)-L-path=$(traversal)")
    MVDA.save_cv_results(partial_filename*".dat", replicates, grids)

    # Default plot options.
    w, h = default(:size)
    options = (; left_margin=5Plots.mm, size=(1.5*w, 1.5*h),)

    # Summarize CV results over folds + make plot.
    @info "Summarizing over folds"
    cv_results = CSV.read(partial_filename*".dat", DataFrame, header=true)
    cv_paths = MVDA.cv_error(cv_results)
    for (title, metric) in zip(titles, metrics)
        fig = MVDA.plot_cv_paths(cv_paths, metric)
        plot!(fig; title=title, options...)
        fig = add_model_size_guide(fig, p)
        savefig(fig, partial_filename*"-replicates=$(metric).png")
    end

    # Construct credible intervals for detailed summary plot.
    @info "Constructing credible intervals"
    cv_intervals = MVDA.credible_intervals(cv_paths)
    for (title, metric) in zip(titles, metrics)
        fig = MVDA.plot_credible_intervals(cv_intervals, metric)
        plot!(fig; title=title, options...)
        fig = add_model_size_guide(fig, p)
        savefig(fig, partial_filename*"-summary=$(metric).png")
    end
    println()

    return nothing
end

# Examples ordered from easiest to hardest
examples = (
    ("colon", 3, 0.8, false),
    ("srbctA", 3, 0.8, false),
    ("leukemiaA", 3, 0.8, false),
    ("lymphomaA", 3, 0.8, false),
    ("brain", 3, 0.8, false),
    ("prostate", 3, 0.8, false),
)

for (example, nfolds, split, shuffle) in examples
    tmp = CSV.read("/home/alanderos/Desktop/data/$(example).DAT", DataFrame, header=false)
    df = shuffle ? shuffleobs(tmp) : tmp[:,:]
    data = (Vector(df[!,end]), Matrix{Float64}(df[!,1:end-1]))
    run("/home/alanderos/Desktop/VDA/", example, data, false;
        at=split,           # CV set / Test set split
        nfolds=nfolds,      # number of folds
        nreplicates=50,     # number of CV replicates
    )
end
