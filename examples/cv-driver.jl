using MVDA, Plots, Statistics, Random, LinearAlgebra, CSV, DataFrames
using ProgressMeter

# helper functions for evaluating misclassification errors
get_average_score(result, i) = map(x -> x[i], dropdims(mean(result.score, dims=3), dims=(2,3)))
function get_std_score(result, i)
    tmp = map(x -> x[i], result.score)
    dropdims(std(tmp, dims=3), dims=(2,3))
end

function run_example(rng::AbstractRNG, filename, title, targets, X, nreplicates::Int, sparse2dense::Bool)
    # Load data and create problem instance.
    X .= (X .- mean(X, dims=1)) ./ std(X, dims=1)

    # Extract problem info and set deadzone radius to maximal value.
    problem = MVDAProblem(targets, X, intercept=true)
    n, p, c = MVDA.probdims(problem)
    ϵ = ifelse(c == 2, 0.5, 0.5 * sqrt(2*c/(c-1)))

    # Create grid for CV.
    ϵ_grid = [ϵ]
    s_grid = [1-k/p for k in p:-1:0]
    if sparse2dense sort!(s_grid, rev=true) end

    # Allocate permutation vector and paths
    idx = similar(Vector{Int}, axes(X, 1))
    cvpath = Vector{Vector{Vector{Float64}}}(undef, nreplicates)
    selected_sparsity = Vector{Float64}(undef, nreplicates)

    # Repeat CV multiple times to estimate statistical properties of errors.
    # println("Example $(dataset); n=$(n), p=$(p), c=$(c)")
    t = @elapsed @showprogress "Example: $(filename) " for r in 1:nreplicates
        # Shuffle data samples.
        randperm!(rng, idx)
        problem.X .= problem.X[idx, :]
        problem.Y .= problem.Y[idx, :]
        targets .= targets[idx]
    
        # Compute initial solution using regularization.
        if sparse2dense # sparse -> dense
            fill!(problem.coeff.all, 0)
            @views copyto!(problem.coeff.all[end, :], mean(problem.Y, dims=1))
        else # dense -> sparse
            fill!(problem.coeff.all, 1/p)
            @views copyto!(problem.coeff.all[end, :], mean(problem.Y, dims=1))
            # fit_regMVDA(MMSVD(), problem, ϵ, 1.0, gtol=1e-12)
        end
    
        # Run 3-fold cross-validation.
        # println("Replicate $(r)")
        result = cv_MVDA(MMSVD(), problem, ϵ_grid, s_grid,
            nouter=10^2,
            inner=10^4,
            nfolds=3,
            gtol=1e-8,
            dtol=1e-6,
            rtol=0.0,
            nesterov_threshold=100,
            progressbar=false,
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
    std_score = map(i -> std(map(x -> x[i], cvpath)), 1:3) ./ sqrt(nreplicates)
    avg_model = mean(selected_sparsity)
    std_model = std(selected_sparsity) ./ sqrt(nreplicates)
    
    # Median CV errors & sparsity over replicates
    med_score = map(i -> [median(map(x -> x[i][j], cvpath)) for j in eachindex(s_grid)], 1:3)
    qt1_score = med_score .- map(i -> [quantile(map(x -> x[i][j], cvpath), 1//4) for j in eachindex(s_grid)], 1:3)
    qt3_score = map(i -> [quantile(map(x -> x[i][j], cvpath), 3//4) for j in eachindex(s_grid)], 1:3) .- med_score
    med_model = median(selected_sparsity)
    qt1_model = med_model - quantile(selected_sparsity, 1//4)
    qt3_model = quantile(selected_sparsity, 3//4) - med_model

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
    foreach(path -> plot!(xs, path, lw=3, color=:black, alpha=1/nreplicates, label=nothing, title=title), cvpath)
    plot!(xs, avg_score[2], lw=3, color=:red, label="mean")
    plot!(xs, med_score[2], lw=3, color=:blue, label="median")
    savefig(fig, "~/Desktop/VDA/$(filename)-errors.png")

    # Plot CV estimates + standard errors as functions of sparsity.
    layout = @layout [a{1e-8h}; grid(3,1)]
    fig = plot(; title=["" "Training" "Validation" "Test"], layout=layout, options...)
    plot!(fig, title=title, framestyle=:none, subplot=1)
    for i in 1:3
        a = avg_model
        j = searchsortedlast(sort(xs), a)
        j = ifelse(sparse2dense, length(xs)-j+1, j)
        b = avg_score[i][j]
        arnd = round(a, sigdigits=3)
        brnd = round(b, sigdigits=3)
        plot!(fig, xs, avg_score[i], ribbon=std_score[i], lw=1, subplot=i+1)
        scatter!(fig, (a, b), xerr=std_model, marker=:cross, color=:black, markersize=8, subplot=i+1)
        annotate!(fig, [(a, brnd+20, ("($(arnd), $(brnd))", 12, :center))], subplot=i+1)
    end
    savefig(fig, "~/Desktop/VDA/$(filename)-summaryA.png")

    # Plot median CV estimates + quantiles as functions of sparsity.
    layout = @layout [a{1e-8h}; grid(3,1)]
    fig = plot(; title=["" "Training" "Validation" "Test"], layout=layout, options...)
    plot!(fig, title=title, framestyle=:none, subplot=1)
    for i in 1:3
        a = med_model
        j = searchsortedlast(sort(xs), a)
        j = ifelse(sparse2dense, length(xs)-j+1, j)
        b = med_score[i][j]
        arnd = round(a, sigdigits=3)
        brnd = round(b, sigdigits=3)
        plot!(fig, xs, med_score[i], ribbon=(qt1_score[i], qt3_score[i]), lw=1, subplot=i+1)
        scatter!(fig, (a, b), xerr=([qt1_model], [qt3_model]), marker=:cross, color=:black, markersize=8, subplot=i+1)
        annotate!(fig, [(a, brnd+20, ("($(arnd), $(brnd))", 12, :center))], subplot=i+1)
    end
    savefig(fig, "~/Desktop/VDA/$(filename)-summaryB.png")

    return t
end
