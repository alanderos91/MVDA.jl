using DataFrames, MLDataUtils, KernelFunctions, MVDA, StableRNGs
using LinearAlgebra, Statistics
using Polyester

# using MKL
BLAS.set_num_threads(8)

run = function(dir, example, algorithm, data, sparse2dense::Bool=false; at::Real=0.8, nfolds::Int=5, kwargs...)
    # Create MVDAProblem instance w/ RBFKernel and construct grids.
    @info "Creating MVDAProblem for $(example)"
    dist = Float64[]
    for j in axes(data[2], 1)
        @views yⱼ, xⱼ = data[1][j], data[2][j, :]
        @batch per=core threadlocal=Float64[]::Vector{Float64} for i in axes(data[2], 1)
            @views yᵢ, xᵢ = data[1][i], data[2][i, :]
            if yᵢ != yⱼ
                push!(threadlocal, norm(xᵢ-xⱼ))
            end
        end
        foreach(subset -> append!(dist, subset), threadlocal)
    end
    σ = median(dist)
    problem = MVDAProblem(data..., intercept=true, kernel=σ*RBFKernel())
    n, p, c = MVDA.probdims(problem)
    n_train = round(Int, n * at * (nfolds-1)/nfolds)
    n_validate = round(Int, n * at * 1/nfolds)
    n_test = round(Int, n * (1-at))
    fill!(problem.coeff.all, 1/(n_train+1))
    fill!(problem.coeff_prev.all, 1/(n_train+1))
    ϵ_grid = [MVDA.maximal_deadzone(problem)]

    if example == "HAR"
        s_grid = sort!(collect(range(0.0, 1.0, length=21)), rev=sparse2dense)
    else
        s_grid = sort!([1-k/n_train for k in n_train:-1:0], rev=sparse2dense)
    end
    grids = (ϵ_grid, s_grid)

    # Collect data for cross-validation replicates.
    @info "CV split: $(n_train) Train / $(n_validate) Validate / $(n_test) Test"
    subsets = (n_train, n_validate, n_test)
    rng = StableRNG(1903)
    replicates = MVDA.cv_estimation(algorithm, problem, grids;
        at=at,                  # propagate CV / Test split
        nfolds=nfolds,          # propagate number of folds
        rng=rng,                # random number generator for reproducibility
        nouter=10^2,            # outer iterations
        ninner=10^6,            # inner iterations
        gtol=1e-3,              # tolerance on gradient for convergence of inner problem
        dtol=1e-3,              # tolerance on distance for convergence of outer problem
        rtol=0.0,               # use strict distance criteria
        lambda=1e-3,            # regularization level
        tol=1e-6,               # convergence tolerance in initialization
        maxiter=10^4,           # maximum iterations in initialization
        nesterov_threshold=100, # delay on Nesterov acceleration
        show_progress=true,     # display progress over replicates
        kwargs...               # propagate other keywords
    )

    # Write results to disk.
    @info "Writing to disk"
    traversal = sparse2dense ? "S2D" : "D2S"
    filename = joinpath(dir, "$(example)-L-path=$(traversal).dat")
    kernel = "RBF"
    problem_info = (example, subsets, p, c, sparse2dense, kernel)
    MVDA.save_cv_results(filename, problem_info, replicates, grids)

    return nothing
end

# Gaussian Clouds
ex1 = function()
    n_cv, n_test = 250, 10^3
    nsamples = n_cv + n_test
    nclasses = 3
    data = MVDA.simulate_gaussian_clouds(nsamples, nclasses; sigma=0.25, rng=StableRNG(1903))
    (MMSVD(), data, "clouds", 5, n_cv / nsamples)
end

# Nested Circles
ex2 = function()
    n_cv, n_test = 250, 10^3
    nsamples = n_cv + n_test
    nclasses = 3
    data = MVDA.simulate_nested_circles(nsamples, nclasses; p=8//10, rng=StableRNG(1903))
    (MMSVD(), data, "circles", 5, n_cv / nsamples)
end

# Waveform
ex3 = function()
    n_cv, n_test = 375, 10^3
    nsamples = n_cv + n_test
    nfeatures = 21
    data = MVDA.simulate_waveform(nsamples, nfeatures; rng=StableRNG(1903))
    (MMSVD(), data, "waveform", 5, n_cv / nsamples)
end

# Zoo
ex4 = function()
    df = MVDA.dataset("zoo")
    data = (Vector(df[!,1]), Matrix{Float64}(df[!,2:end]))
    (MMSVD(), data, "zoo", 3, 0.9)
end

# Vowel
ex5 = function()
    df = MVDA.dataset("vowel")
    data = (Vector(df[!,1]), Matrix{Float64}(df[!,2:end]))
    (MMSVD(), data, "vowel", 5, 528 / 990)
end

# HAR
# ex6 = function()
#     df = MVDA.dataset("HAR")
#     data = (Vector(df[!,1]), Matrix{Float64}(df[!,2:end]))
#     (SD(), data, "HAR", 5, 7352 / 10299)
# end

examples = (ex1, ex2, ex3, ex4, ex5,)
dir = ARGS[1]
@info "Output directory: $(dir)"

for f in examples
    algorithm, data, example, nfolds, split = f()
    run(dir, example, algorithm, data, false;
        at=split,           # CV set / Test set split
        nfolds=nfolds,      # number of folds
        nreplicates=50,     # number of CV replicates
    )
end
