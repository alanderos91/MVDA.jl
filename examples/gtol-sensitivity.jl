using MVDA, Statistics, Random, LinearAlgebra, ProgressMeter, Plots

# helper functions for evaluating misclassification errors
function get_average_score(result, i)
    tmp = map(x -> x[i], result.score)
    dropdims(mean(tmp, dims=3), dims=(2,3))
end

function get_std_score(result, i)
    tmp = map(x -> x[i], result.score)
    dropdims(std(tmp, dims=3), dims=(2,3))
end

function gtol_sensitivity(tol, data)
    # Standardize data and create problem instance with intercept.
    targets, _X = data
    idx = similar(Vector{Int}, axes(_X, 1))
    randperm!(MersenneTwister(1234), idx)
    targets, X = targets[idx], _X[idx, :]
    X .= (X .- mean(X, dims=1)) ./ std(X, dims=1)
    problem = MVDAProblem(targets, X, intercept=true)
    
    # Create problem and create grids.
    _, p, c = MVDA.probdims(problem)
    ϵ = ifelse(c == 2, 0.5, 0.5 * sqrt(2*c/(c-1)))
    ϵ_grid = [ϵ]
    s_grid = [1-k/p for k in p:-1:0]

    # Initialize and run 5-fold cross-validiation.
    fill!(problem.coeff.all, 1/p)
    @views copyto!(problem.coeff.all[end, :], mean(problem.Y, dims=1))
    
    result = cv_MVDA(MMSVD(), problem, ϵ_grid, s_grid;
        progressbar=false,
        gtol=tol,
        nouter=10^2,
        ninner=10^5,
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

p = 100
c = 4
d = 3.0
nsamples1 = 20
nsamples2 = 500
underdetermined_data = MVDA.simulate_WS2007(nsamples1*c, p, c, nsamples1, d)
overdetermined_data = MVDA.simulate_WS2007(nsamples2*c, p, c, nsamples2, d)

underdetermined(tol) = gtol_sensitivity(tol, underdetermined_data)
overdetermined(tol) = gtol_sensitivity(tol, overdetermined_data)

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
    plot!(path, gtol, lw=3, ls=:dot, alpha=0.5, color=:white, legend=nothing, framestyle=:grid, grid=nothing)
    
    return fig
end

yticks = xs[1:3:npoints]
plot_summary(xs, ys[1].V, ys[1].s, p, yticks=yticks, dpi=300); savefig("~/Desktop/VDA/gtol-underdetermined.png")
plot_summary(xs, ys[2].V, ys[2].s, p, yticks=yticks, dpi=300); savefig("~/Desktop/VDA/gtol-overdetermined.png")
