using MVDA, MLDataUtils, Statistics, StatsBase, LinearAlgebra
using Random, StableRNGs
using ProgressMeter, Plots

# helper functions for evaluating misclassification errors
function get_average_score(result, i)
    tmp = map(x -> x[i], result.score)
    dropdims(mean(tmp, dims=3), dims=(2,3))
end

function get_std_score(result, i)
    tmp = map(x -> x[i], result.score)
    dropdims(std(tmp, dims=3), dims=(2,3))
end

function standardize!(X)
    F = fit(ZScoreTransform, X, dims=1)
    StatsBase.transform!(F, X)
end

_permute!(x::AbstractVector, idx) = permute!(x, idx)
_permuterows!(X::AbstractMatrix, idx) = foreach(row -> _permute!(row, idx), eachrow(X))
_permutecols!(X::AbstractMatrix, idx) = foreach(col -> _permute!(col, idx), eachcol(X))

function make_vertex_matrix!(Y, problem, labels)
    label2vertex = problem.label2vertex
    for (i, label_i) in enumerate(labels)
        Y[i,:] .= label2vertex[label_i]
    end
    return Y
end

function gtol_sensitivity(tol, cv_data, test_data)
    # Standardize data and create problem instance with intercept.
    targets, X = cv_data
    problem = MVDAProblem(targets, X, intercept=true)

    # Get problem size and create grids.
    n, p, c = MVDA.probdims(problem)
    ϵ = ifelse(c == 2, 0.5, 0.5 * sqrt(2*c/(c-1)))
    ϵ_grid = [ϵ]
    s_grid = [1-k/p for k in p:-1:0]

    # Create CV and Test sets based on vertex encoding.
    ntest = length(test_data[1])
    CV_Y = make_vertex_matrix!(Matrix{Float64}(undef, n, c-1), problem, cv_data[1])
    CV_X = problem.X
    T_Y  = make_vertex_matrix!(Matrix{Float64}(undef, ntest, c-1), problem, test_data[1])
    T_X  = problem.intercept ? [test_data[2] ones(ntest)] : test_data[2]

    # Initialize and run 5-fold cross-validiation.
    fill!(problem.coeff.all, 1/p)
    @views copyto!(problem.coeff.all[end, :], mean(problem.Y, dims=1))
    
    result = cv_MVDA(MMSVD(), problem, (CV_Y, CV_X), (T_Y, T_X), ϵ_grid, s_grid;
        progressbar=false,
        gtol=tol,
        nouter=10^2,
        ninner=10^6,
        nfolds=5,
        dtol=1e-6,
        rtol=0.0,
        nesterov_threshold=100,
    )

    # Get estimates of training, validation, and test set errors.
    avg_fold_score = map(i -> get_average_score(result, i), 1:3)

    # Find model maximizing parsimony and minimizing errors on the basis of validation errors.
    adjusted_score = [(score, 100*(1-sparsity)) for (sparsity, score) in zip(s_grid, avg_fold_score[2])]
    s_optimal = 100 * s_grid[argmin(adjusted_score)]

    return s_optimal, avg_fold_score
end

# Simulation settings
p = 100
c = 4
d = 3.0

function simulate(n, rng)
    global p, c, d
    targets, X = MVDA.simulate_WS2007(n*c, p, c, n, d; rng=rng)
    standardize!(X)
    return (targets, X)
end

nsamples1 = 20
nsamples2 = 500
seed = 1122
rng = StableRNG(0)

Random.seed!(rng, 1122)
underdetermined_data = splitobs(simulate(nsamples1, rng), at=0.8, obsdim=1)

Random.seed!(rng, 1122)
overdetermined_data = splitobs(simulate(nsamples2, rng), at=0.8, obsdim=1)

underdetermined(tol) = gtol_sensitivity(tol, underdetermined_data...)
overdetermined(tol) = gtol_sensitivity(tol, overdetermined_data...)

alloc_path(n) = Vector{Vector{Float64}}(undef, n)
npoints = 25
xs = 10.0 .^ range(-8, 0, length=npoints)
ys = [
        (;s=zeros(npoints), Tr=alloc_path(npoints), V=alloc_path(npoints), T=alloc_path(npoints)),
        (;s=zeros(npoints), Tr=alloc_path(npoints), V=alloc_path(npoints), T=alloc_path(npoints))
    ]

pbar = Progress(2*npoints, 1, "Running... ")
for (i, x) in enumerate(xs)
    (s, (Tr, V, T)) = underdetermined(x)
    ys[1].s[i] = s
    ys[1].Tr[i] = Tr
    ys[1].V[i] = V
    ys[1].T[i] = T
    next!(pbar)

    (s, (Tr, V, T)) = overdetermined(x)
    ys[2].s[i] = s
    ys[2].Tr[i] = Tr
    ys[2].V[i] = V
    ys[2].T[i] = T
    next!(pbar)
end

function plot_summary(gtol, errors, path, p; kwargs...)
    x = [100*k/p for k in 0:p]
    y = gtol
    z = vcat(errors'...)
    
    fig = heatmap(x, y, z, yscale=:log10; xlabel="Sparsity (%)", ylabel="gtol", colorbar_title="Classification Error (%)", clim=(0,100), color=:vik, kwargs...)
    plot!(path, gtol, lw=3, ls=:dot, alpha=0.75, color=:white, legend=nothing, framestyle=:grid, grid=nothing)
    
    return fig
end

yticks = xs[1:3:npoints]
plot_summary(xs, ys[1].Tr, ys[1].s, p, yticks=yticks, dpi=300); savefig("~/Desktop/VDA/gtol-underdetermined-Tr.png")
plot_summary(xs, ys[1].V, ys[1].s, p, yticks=yticks, dpi=300); savefig("~/Desktop/VDA/gtol-underdetermined-V.png")
plot_summary(xs, ys[1].T, ys[1].s, p, yticks=yticks, dpi=300); savefig("~/Desktop/VDA/gtol-underdetermined-T.png")

plot_summary(xs, ys[2].Tr, ys[2].s, p, yticks=yticks, dpi=300); savefig("~/Desktop/VDA/gtol-overdetermined-Tr.png")
plot_summary(xs, ys[2].V, ys[2].s, p, yticks=yticks, dpi=300); savefig("~/Desktop/VDA/gtol-overdetermined-V.png")
plot_summary(xs, ys[2].T, ys[2].s, p, yticks=yticks, dpi=300); savefig("~/Desktop/VDA/gtol-overdetermined-T.png")

