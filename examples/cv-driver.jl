using MVDA, Plots, Statistics, StatsBase, Random, LinearAlgebra, CSV, DataFrames
using ProgressMeter

function subheader(str)
    join(map(s -> string(str,"_",s), ("avg", "std", "med", "clo", "chi")), ',')
end

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

function run_example(rng::AbstractRNG, filename, title, cv_set, test_set, nreplicates::Int, s_grid, sparse2dense::Bool; kwargs...)
    # Create problem and set dead zone radius to maximal value.
    problem = MVDAProblem(cv_set[1], cv_set[2], intercept=true)
    n, p, c = MVDA.probdims(problem)
    ϵ = ifelse(c == 2, 0.5, 0.5 * sqrt(2*c/(c-1)))

    ntest = length(test_set[1])
    CV_Y = make_vertex_matrix!(Matrix{Float64}(undef, n, c-1), problem, cv_set[1])
    CV_X = problem.X
    T_Y  = make_vertex_matrix!(Matrix{Float64}(undef, ntest, c-1), problem, test_set[1])
    T_X  = problem.intercept ? [test_set[2] ones(ntest)] : test_set[2]

    # Create grid for CV.
    ϵ_grid = [ϵ]
    sort!(s_grid, rev=sparse2dense)

    # Allocate permutation vector and paths
    idx = Vector{Int}(undef, n)
    cvpath = Vector{Vector{Vector{Float64}}}(undef, nreplicates)
    selected_sparsity = Vector{Float64}(undef, nreplicates)

    # Repeat CV multiple times to estimate statistical properties of errors.
    t = @elapsed @showprogress "Example: $(filename) " for r in 1:nreplicates
        # Shuffle data samples.
        randperm!(rng, idx)
        _permutecols!(CV_Y, idx)
        _permutecols!(CV_X, idx)
    
        # Compute initial solution using regularization.
        if sparse2dense # sparse -> dense
            fill!(problem.coeff.all, 0)
            @views copyto!(problem.coeff.all[end, :], mean(problem.Y, dims=1))
        else # dense -> sparse
            fill!(problem.coeff.all, 1/p)
            @views copyto!(problem.coeff.all[end, :], mean(problem.Y, dims=1))
        end
    
        # Run cross-validation.
        result = cv_MVDA(MMSVD(), problem, (CV_Y, CV_X), (T_Y, T_X), ϵ_grid, s_grid;
            progressbar=false, kwargs...
        )

        # Get estimates of training, validation, and test set errors.
        avg_fold_score = map(i -> get_average_score(result, i), 1:3)

        # Find model maximizing parsimony and minimizing errors on the basis of validation errors.
        adjusted_score = [(score, 100*(1-sparsity)) for (sparsity, score) in zip(s_grid, avg_fold_score[2])]
        s_optimal = s_grid[argmin(adjusted_score)]

        # Record the path and optimal model.
        cvpath[r] = avg_fold_score
        selected_sparsity[r] = 100 * s_optimal
    end

    # Average CV errors & sparsity over replicates.
    avg_score = mean(cvpath)
    std_score = map(i -> std(map(x -> x[i], cvpath)), 1:3)
    avg_model = mean(selected_sparsity)
    std_model = std(selected_sparsity)
    
    # Median CV errors & sparsity over replicates
    credibility = 19//20
    α = (1-credibility)/2

    med_score = map(i -> [median(map(x -> x[i][j], cvpath)) for j in eachindex(s_grid)], 1:3)
    clo_score = map(i -> [quantile(map(x -> x[i][j], cvpath), α) for j in eachindex(s_grid)], 1:3)
    chi_score = map(i -> [quantile(map(x -> x[i][j], cvpath), 1-α) for j in eachindex(s_grid)], 1:3)
    med_model = median(selected_sparsity)
    clo_model = quantile(selected_sparsity, α)
    chi_model = quantile(selected_sparsity, 1-α)

    # Default options.
    xs = 100 .* s_grid
    w, h = default(:size)
    options = (;
        xlabel="Sparsity (%)",
        ylabel="Error (%)",
        legend=nothing,
        xlims=(0,110),
        ylims=(-1,101),
        xticks=0:10:100,
        yticks=0:20:100,
        left_margin=5Plots.mm,
        size=(1.5*w, 1.5*h),
    )

    # Plot each CV validation path as a function of sparsity with mean trajectory highlighted in red; median in blue.
    fig = plot(;options...)
    foreach(path -> plot!(xs, path[2], lw=3, color=:black, alpha=1/sqrt(nreplicates), label=nothing, title=title), cvpath)
    plot!(xs, avg_score[2], lw=3, color=:red, label="mean")
    plot!(xs, med_score[2], lw=3, color=:blue, label="median")
    savefig(fig, "~/Desktop/VDA/$(filename)-errors.png")

    # Plot CV estimates + standard errors as functions of sparsity.
    layout = @layout [a{1e-8h}; grid(3,1)]
    fig = plot(; title=["" "Training" "Validation" "Test"], layout=layout, options...)
    plot!(fig, title=title, framestyle=:none, subplot=1)
    for i in 1:3
        a = avg_model
        j = ifelse(sparse2dense, findfirst(≤(a), xs), findlast(≤(a), xs))
        b = avg_score[i][j]
        arnd = round(a, sigdigits=4)
        brnd = round(b, sigdigits=4)
        plot!(fig, xs, avg_score[i], ribbon=std_score[i], lw=1, subplot=i+1)
        scatter!(fig, (a, b), xerr=std_model, marker=:x, color=:black, markersize=8, subplot=i+1)
        annotate!(fig, [(a, brnd+20, ("($(arnd), $(brnd))", 10, :center))], subplot=i+1)
    end
    savefig(fig, "~/Desktop/VDA/$(filename)-summaryA.png")

    # Plot median CV estimates + quantiles as functions of sparsity.
    layout = @layout [a{1e-8h}; grid(3,1)]
    fig = plot(; title=["" "Training" "Validation" "Test"], layout=layout, options...)
    plot!(fig, title=title, framestyle=:none, subplot=1)
    for i in 1:3
        a = med_model
        j = ifelse(sparse2dense, findfirst(≤(a), xs), findlast(≤(a), xs))
        b = med_score[i][j]
        arnd = round(a, sigdigits=4)
        brnd = round(b, sigdigits=4)
        lower = med_score[i] .- clo_score[i]
        upper = chi_score[i] .- med_score[i]
        error_bars = [(a - clo_model, chi_model - a)]
        plot!(fig, xs, med_score[i], ribbon=(lower, upper), lw=1, subplot=i+1)
        scatter!(fig, (a, b), xerr=error_bars, marker=:x, color=:black, markersize=8, subplot=i+1)
        annotate!(fig, [(a, brnd+20, ("($(arnd), $(brnd))", 10, :center))], subplot=i+1)
    end
    savefig(fig, "~/Desktop/VDA/$(filename)-summaryB.png")

    sparsity = (;
        avg=avg_model,
        std=std_model,
        med=med_model,
        clo=clo_model,
        chi=chi_model,
    )

    j_avg = ifelse(sparse2dense, findfirst(≤(avg_model), xs), findlast(≤(avg_model), xs))
    j_med = ifelse(sparse2dense, findfirst(≤(med_model), xs), findlast(≤(med_model), xs))
    train, validation, test = map(
        i -> (;
            avg=avg_score[i][j_avg],
            std=std_score[i][j_avg],
            med=med_score[i][j_med],
            clo=clo_score[i][j_med],
            chi=chi_score[i][j_med],
        ),
        1:3
    )
    
    return t, sparsity, train, validation, test
end

function run_nonlinear_example(rng::AbstractRNG, filename, title, kernel, cv_set, test_set, nreplicates::Int, s_grid, sparse2dense::Bool; kwargs...)
    # Create problem and set dead zone radius to maximal value.
    problem = NonLinearMVDAProblem(kernel, cv_set[1], cv_set[2], intercept=true)
    n, p, c = MVDA.probdims(problem)
    ϵ = ifelse(c == 2, 0.5, 0.5 * sqrt(2*c/(c-1)))

    ntest = length(test_set[1])
    CV_Y = make_vertex_matrix!(Matrix{Float64}(undef, n, c-1), problem, cv_set[1])
    CV_X = problem.X
    CV_K = view(problem.K, 1:n, 1:n)
    T_Y  = make_vertex_matrix!(Matrix{Float64}(undef, ntest, c-1), problem, test_set[1])
    T_X  = test_set[2]

    # Compute initial solution using regularization.
    if sparse2dense # sparse -> dense
        fill!(problem.coeff.all, 0)
        @views copyto!(problem.coeff.all[end, :], mean(problem.Y, dims=1))
    else # dense -> sparse
        fill!(problem.coeff.all, 1/n)
        @views copyto!(problem.coeff.all[end, :], mean(problem.Y, dims=1))
    end

    # Create grid for CV.
    ϵ_grid = [ϵ]
    sort!(s_grid, rev=sparse2dense)

    # Allocate permutation vector and paths
    idx = Vector{Int}(undef, n)
    cvpath = Vector{Vector{Vector{Float64}}}(undef, nreplicates)
    selected_sparsity = Vector{Float64}(undef, nreplicates)

    # Repeat CV multiple times to estimate statistical properties of errors.
    t = @elapsed @showprogress "Example: $(filename) " for r in 1:nreplicates
        # Shuffle data samples.
        randperm!(rng, idx)
        _permutecols!(CV_Y, idx)
        _permutecols!(CV_X, idx)
    
        # Run cross-validation.
        result = cv_MVDA(MMSVD(), problem, (CV_Y, CV_X), (T_Y, T_X), ϵ_grid, s_grid;
            progressbar=false, kwargs...
        )

        # Get estimates of training, validation, and test set errors.
        avg_fold_score = map(i -> get_average_score(result, i), 1:3)

        # Find model maximizing parsimony and minimizing errors on the basis of validation errors.
        adjusted_score = [(score, 100*(1-sparsity)) for (sparsity, score) in zip(s_grid, avg_fold_score[2])]
        s_optimal = s_grid[argmin(adjusted_score)]

        # Record the path and optimal model.
        cvpath[r] = avg_fold_score
        selected_sparsity[r] = 100 * s_optimal
    end

    # Average CV errors & sparsity over replicates.
    avg_score = mean(cvpath)
    std_score = map(i -> std(map(x -> x[i], cvpath)), 1:3)
    avg_model = mean(selected_sparsity)
    std_model = std(selected_sparsity)
    
    # Median CV errors & sparsity over replicates
    credibility = 19//20
    α = (1-credibility)/2

    med_score = map(i -> [median(map(x -> x[i][j], cvpath)) for j in eachindex(s_grid)], 1:3)
    clo_score = map(i -> [quantile(map(x -> x[i][j], cvpath), α) for j in eachindex(s_grid)], 1:3)
    chi_score = map(i -> [quantile(map(x -> x[i][j], cvpath), 1-α) for j in eachindex(s_grid)], 1:3)
    med_model = median(selected_sparsity)
    clo_model = quantile(selected_sparsity, α)
    chi_model = quantile(selected_sparsity, 1-α)

    # Default options.
    xs = 100 .* s_grid
    w, h = default(:size)
    options = (;
        xlabel="Sparsity (%)",
        ylabel="Error (%)",
        legend=nothing,
        xlims=(0,110),
        ylims=(-1,101),
        xticks=0:10:100,
        yticks=0:20:100,
        left_margin=5Plots.mm,
        size=(1.5*w, 1.5*h),
    )

    # Plot each CV validation path as a function of sparsity with mean trajectory highlighted in red; median in blue.
    fig = plot(;options...)
    foreach(path -> plot!(xs, path[2], lw=3, color=:black, alpha=1/nreplicates, label=nothing, title=title), cvpath)
    plot!(xs, avg_score[2], lw=3, color=:red, label="mean")
    plot!(xs, med_score[2], lw=3, color=:blue, label="median")
    savefig(fig, "~/Desktop/VDA/$(filename)-errors.png")

    # Plot CV estimates + standard errors as functions of sparsity.
    layout = @layout [a{1e-8h}; grid(3,1)]
    fig = plot(; title=["" "Training" "Validation" "Test"], layout=layout, options...)
    plot!(fig, title=title, framestyle=:none, subplot=1)
    for i in 1:3
        a = avg_model
        j = ifelse(sparse2dense, findfirst(≤(a), xs), findlast(≤(a), xs))
        b = avg_score[i][j]
        arnd = round(a, sigdigits=4)
        brnd = round(b, sigdigits=4)
        plot!(fig, xs, avg_score[i], ribbon=std_score[i], lw=1, subplot=i+1)
        scatter!(fig, (a, b), xerr=std_model, marker=:x, color=:black, markersize=8, subplot=i+1)
        annotate!(fig, [(a, brnd+20, ("($(arnd), $(brnd))", 10, :center))], subplot=i+1)
    end
    savefig(fig, "~/Desktop/VDA/$(filename)-summaryA.png")

    # Plot median CV estimates + quantiles as functions of sparsity.
    layout = @layout [a{1e-8h}; grid(3,1)]
    fig = plot(; title=["" "Training" "Validation" "Test"], layout=layout, options...)
    plot!(fig, title=title, framestyle=:none, subplot=1)
    for i in 1:3
        a = med_model
        j = ifelse(sparse2dense, findfirst(≤(a), xs), findlast(≤(a), xs))
        b = med_score[i][j]
        arnd = round(a, sigdigits=4)
        brnd = round(b, sigdigits=4)
        lower = med_score[i] .- clo_score[i]
        upper = chi_score[i] .- med_score[i]
        error_bars = [(a - clo_model, chi_model - a)]
        plot!(fig, xs, med_score[i], ribbon=(lower, upper), lw=1, subplot=i+1)
        scatter!(fig, (a, b), xerr=error_bars, marker=:x, color=:black, markersize=8, subplot=i+1)
        annotate!(fig, [(a, brnd+20, ("($(arnd), $(brnd))", 10, :center))], subplot=i+1)
    end
    savefig(fig, "~/Desktop/VDA/$(filename)-summaryB.png")

    sparsity = (;
        avg=avg_model,
        std=std_model,
        med=med_model,
        clo=clo_model,
        chi=chi_model,
    )

    j_avg = ifelse(sparse2dense, findfirst(≤(avg_model), xs), findlast(≤(avg_model), xs))
    j_med = ifelse(sparse2dense, findfirst(≤(med_model), xs), findlast(≤(med_model), xs))
    train, validation, test = map(
        i -> (;
            avg=avg_score[i][j_avg],
            std=std_score[i][j_avg],
            med=med_score[i][j_med],
            clo=clo_score[i][j_med],
            chi=chi_score[i][j_med],
        ),
        1:3
    )

    return t, sparsity, train, validation, test
end
