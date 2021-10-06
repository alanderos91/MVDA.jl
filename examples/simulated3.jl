using MVDA, Plots, Statistics, Random, LinearAlgebra

function draw_circle!(v, ϵ)
    xs = v[1] .+ range(-ϵ, ϵ, length=201)
    ys1 = sqrt.(ϵ^2 .- (xs .- v[1]) .^2) .+ v[2]
    ys2 = -sqrt.(ϵ^2 .- (xs .- v[1]) .^2) .+ v[2]
    plot!(xs, ys1, color=:black, style=:dash, label="")
    plot!(xs, ys2, color=:black, style=:dash, label="")
end

function simulate(n, p, c, nsamples, overlap)
    # simulate 3 classes with single predictor as in Wang and Shen 2007
    X = randn(n, p)
    for i in 1:c
        idx = nsamples*(i-1)+1:nsamples*i
        a₁ = overlap * cos(2*(π/6 + (i-1)*π/c))
        a₂ = overlap * sin(2*(π/6 + (i-1)*π/c))
        X[idx,1] .+= a₁
        X[idx,2] .+= a₂
    end
    targets = zeros(Int, n)
    for j in 1:c
        targets[nsamples*(j-1)+1:nsamples*j] .= j
    end
    idx = randperm(n)
    targets, X = targets[idx], X[idx,:]
    return targets, X
end

function discovery_metrics(x, y)
    TP = FP = TN = FN = 0
    for (xi, yi) in zip(x, y)
        TP += (xi != 0) && (yi != 0)
        FP += (xi != 0) && (yi == 0)
        TN += (xi == 0) && (yi == 0)
        FN += (xi == 0) && (yi != 0)
    end
    return (TP, FP, TN, FN)
end

function compute_FDR(TP, FP, TN, FN)
    v = FP / (TP + FP)
    return isnan(v) ? zero(v) : v
end

function compute_FNR(TP, FP, TN, FN)
    v = FN / (TN + FN)
    return isnan(v) ? zero(v) : v
end

function run_experiment(n, p, c, nsamples)
    ϵ = 0.5 * sqrt(2*c/(c-1))
    s_grid = range(0.0, 0.98, length=15)
    B0 = zeros(p, c-1)
    B0[1:2, :] .= 1
    d_vals = (1.0, 2.0, 3.0)

    m1, m2 = length(s_grid), length(d_vals)
    train_error = zeros(m1, m2); train_error_std = zeros(m1, m2)
    FDR = zeros(m1, m2); FDR_std = zeros(m1, m2)
    FNR = zeros(m1, m2); FNR_std = zeros(m1, m2)

    N = 50
    tmp1 = zeros(m1, N)
    tmp2 = zeros(m1, N)
    tmp3 = zeros(m1, N)

    for (j, d) in enumerate(d_vals)
        for ii in 1:N
            targets, X = simulate(n, p, c, nsamples, d)

            problem = MVDAProblem(targets, X, intercept=true)
            extras = MVDA.__mm_init__(MMSVD(), problem, nothing)
            
            fill!(problem.coeff.all, 1/(p+1))
            fit_regMVDA(MMSVD(), problem, ϵ, 1.0, gtol=1e-8)

            for (i, s) in enumerate(s_grid)
                fit_MVDA!(MMSVD(), problem, ϵ, s, extras, (true,true), verbose=false, nouter=1000, ninner=10^6, rtol=0.0, gtol=1e-6, dtol=1e-6, nesterov_threshold=100);
                Y = problem.Y
                B = problem.proj.all
                V = unique(Y, dims=1)

                labels = MVDA.classify(problem, problem.X)
                metrics = @views discovery_metrics(norm.(eachrow(B[1:p,:])), norm.(eachrow(B0)))

                tmp1[i,ii] = round(100 * (1 - sum(labels .== targets)/n), sigdigits=4)
                tmp2[i,ii] = 100 * compute_FDR(metrics...)
                tmp3[i,ii] = 100 * compute_FNR(metrics...)
            end
        end

        train_error[:,j] .= mean(tmp1, dims=2) |> vec
        train_error_std[:,j] .= std(tmp1, dims=2) ./ N |> vec

        FDR[:,j] .= mean(tmp2, dims=2) |> vec
        FDR_std[:,j] .= std(tmp2, dims=2) ./ N |> vec

        FNR[:,j] .= mean(tmp3, dims=2) |> vec
        FNR_std[:,j] .= std(tmp3, dims=2) ./ N |> vec
    end

    # plot misclassification error as function of sparsity
    w, h = default(:size)
    ms = 2 * default(:markersize)
    fig = plot(xlabel="Sparsity (%)", xlims=(-5,105), ylims=(-5,105), layout=grid(3,1), size=(w,2*h), left_margin=5Plots.mm)
    foreach(i -> vline!([98.0], color=:red, lw=3, style=:dash, label="", subplot=i), 1:3)

    scatter!(100 .* s_grid, train_error,
        yerr=train_error_std,
        title="n=$(n), p=$(p), c=$(c), nⱼ=$(nsamples)",
        ylabel="Error (%)",
        label=reshape(["d = $(d)" for d in d_vals], 1, length(d_vals)),
        marker=[:circ :square :utriangle :dtriangle],
        markersize=ms,
        legend=:topleft,
        lw=3,
        subplot=1,
    )

    scatter!(100 .* s_grid, FDR,
        yerr=FDR_std,
        ylabel="FDR (%)",
        label=reshape(["d = $(d)" for d in d_vals], 1, length(d_vals)),
        marker=[:circ :square :utriangle :dtriangle],
        markersize=ms,
        legend=:bottomleft,
        lw=3,
        subplot=2,
    )

    scatter!(100 .* s_grid, FNR,
        yerr=FNR_std,
        ylabel="FNR (%)",
        label=reshape(["d = $(d)" for d in d_vals], 1, length(d_vals)),
        marker=[:circ :square :utriangle :dtriangle],
        markersize=ms,
        legend=:topleft,
        lw=3,
        subplot=3,
    )

    return fig
end

# underdetermined
nsamples = 20
c = 4
n = nsamples*c
p = 100
fig = run_experiment(n, p, c, nsamples)
savefig(fig, "~/Desktop/simulated3A.png")

# overdetermined
nsamples = 500
c = 4
n = nsamples*c
p = 100
fig = run_experiment(n, p, c, nsamples)
savefig(fig, "~/Desktop/simulated3B.png")
