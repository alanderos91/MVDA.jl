# ---
# title: Benchmarks for MVDA Algorithms
# description: 
# date: 2023-04-26
# author: "[Alfonso Landeros](mailto:alanderos@ucla.edu)"
# ---

# # Benchmarks for MVDA Algorithms

# ## Packages
using MVDA
using StableRNGs
using CairoMakie
import Pkg, InteractiveUtils

# ## Dependencies
Pkg.status()

# ## Environment
InteractiveUtils.versioninfo()

# ## Benchmark Setup

struct BenchmarkRunner
    L
    X
    epsilon_standard
    epsilon_projected
    lambda
    max_outer_iterations
    max_inner_iterations
    acceleration_delay
    projection
end

function make_benchmark_runner(name;
    lambda::Real=1.0,
    max_outer_iterations::Int=100,
    max_inner_iterations::Int=10^4,
    acceleration_delay::Int=10,
    projection::Type=HomogeneousL0Projection,
)
    df = MVDA.dataset(name)
    (L, X) = Vector{String}(df[!,1]), Matrix{Float64}(df[!,2:end])
    nclasses = length(unique(L))
    epsilon_standard = sqrt(2)/2
    epsilon_projected = nclasses == 2 ? 1.0 : 1//2 * sqrt(2*nclasses/(nclasses-1))

    return BenchmarkRunner(L, X,
        epsilon_standard, epsilon_projected, lambda,
        max_outer_iterations, max_inner_iterations, acceleration_delay,
        projection
    )
end

function (F::BenchmarkRunner)(algorithm, encoding, s, (dtol, rtol, gtol), seed)
    #= Create problem instance with specified encoding. =#
    problem = MVDAProblem(F.L, F.X;
        encoding=encoding,
        kernel=nothing,
        intercept=true
    )

    if encoding == :standard
        epsilon = F.epsilon_standard
    elseif encoding == :projected
        epsilon = F.epsilon_projected
    end

    #= Set model coefficients equal to 1. =#
    foreach(
        Base.Fix2(fill!, 1.0),
        problem.coeff
    )

    #= Solve with requested algorithm + settings. =#
    hcb = MVDA.HistoryCallback()
    MVDA.add_field!(hcb,
        :rho, :iters, :risk, :loss, :objective, :gradient, :distance, :penalty)

    @elapsed MVDA.solve_constrained!(MMSVD(), problem, epsilon, F.lambda, s;
        projection_type=F.projection,
        maxrhov=3,
        maxiter=11,
        dtol=dtol,
        rtol=rtol,
        gtol=gtol,
        nesterov=F.acceleration_delay,
        callback=hcb,
        rng=StableRNG(seed),
    )

    hcb = MVDA.HistoryCallback()
    MVDA.add_field!(hcb,
        :rho, :iters, :risk, :loss, :objective, :gradient, :distance, :penalty)

    wall_time = @elapsed MVDA.solve_constrained!(algorithm, problem, epsilon, F.lambda, s;
        projection_type=F.projection,
        maxrhov=F.max_outer_iterations,
        maxiter=F.max_inner_iterations,
        dtol=dtol,
        rtol=rtol,
        gtol=gtol,
        nesterov=F.acceleration_delay,
        callback=hcb,
        rng=StableRNG(seed),
    )

    #= Assess prediction accuracy after fitting model. =#
    accuracy = MVDA.accuracy(problem, (F.L, F.X))

    return (;
        wall_time=wall_time,
        accuracy=accuracy,
        history=hcb.data,
        algorithm=algorithm,
        encoding=encoding,
        sparsity=s,
        dtol=dtol,
        rtol=rtol,
        gtol=gtol,
        seed=seed,
        coeff=problem.coeff,
        coeff_proj=problem.coeff_proj,
    )
end


struct IntegerTicks end

Makie.get_tickvalues(::IntegerTicks, vmin, vmax) = ceil(Int, vmin) : floor(Int, vmax)

# Convergence history

function plot_history(result)
    colors = Makie.wong_colors()
    titles = ["Projected Simplex", "Standard Simplex"]
    labels = ["MMSVD", "SD", "PGD"]
    encodings = ["projected", "standard"]

    fig = Figure()
    ax = [Axis(fig[i,1];
            xscale=log10,
            yscale=log10,
            title=titles[i],
            xlabel="log10[iteration]",
            ylabel="log10[loss]",
            xminorticksvisible=true,
            yminorticksvisible=true,
            xminorticks=IntervalsBetween(9),
            yminorticks=IntervalsBetween(9),
        ) for i in 1:2
    ]

    for (i, enc) in enumerate(encodings)
        for (j, alg) in enumerate(labels)
            r = result[enc][alg]
            lines!(ax[i], r.history[:loss], label=alg, color=colors[j], linewidth=5,)
        end
    end

    elements = [PolyElement(polycolor=colors[j]) for j in eachindex(labels)]
    Legend(fig[1:2,2], elements, labels, "Algorithm")

    return fig
end

# Wall time

function plot_wall_time(result)
    colors = Makie.wong_colors()
    fig = Figure()
    ax = Axis(fig[1,1];
        xlabel="Algorithm",
        ylabel="Time [s]",
        xticks=(1:3, ["MMSVD", "SD", "PGD"]),
    )

    tbl = (x=Int[], y=Float64[], grp=Int[])
    for (x, alg) in enumerate(("MMSVD", "SD", "PGD")), (k, enc) in enumerate(("projected", "standard"))
        push!(tbl.x, x)
        push!(tbl.grp, k)
        push!(tbl.y, result[enc][alg].wall_time)
    end

    barplot!(ax, tbl.x, tbl.y,
        bar_labels=:y,
        dodge=tbl.grp,
        color=colors[tbl.grp]
    )

    labels = ["Projected Simplex", "Standard Simplex"]
    elements = [PolyElement(polycolor=colors[i]) for i in 1:2]
    title = "Vertex Encoding"

    Legend(fig[1,2], elements, labels, title)

    return fig
end

# Prediction Accuracy

function plot_accuracy(result)
    colors = Makie.wong_colors()
    fig = Figure()
    ax = Axis(fig[1,1];
        limits=(nothing, nothing, 0.0, 1.0),
        xlabel="Algorithm",
        ylabel="Accuracy",
        xticks=(1:3, ["MMSVD", "SD", "PGD"]),
        yticks=range(0, 1, length=11),
    )

    tbl = (x=Int[], y=Float64[], grp=Int[])
    for (x, alg) in enumerate(("MMSVD", "SD", "PGD")), (k, enc) in enumerate(("projected", "standard"))
        push!(tbl.x, x)
        push!(tbl.grp, k)
        push!(tbl.y, result[enc][alg].accuracy)
    end

    barplot!(ax, tbl.x, tbl.y,
        bar_labels=:y,
        flip_labels_at=0.8,
        color_over_bar=:white,
        dodge=tbl.grp,
        color=colors[tbl.grp],
    )

    labels = ["Projected Simplex", "Standard Simplex"]
    elements = [PolyElement(polycolor=colors[i]) for i in 1:2]
    title = "Vertex Encoding"

    Legend(fig[1,2], elements, labels, title)

    return fig
end

# ### Iris Dataset

seed = 1903
iris_benchmark = make_benchmark_runner("iris";
    lambda=1.0,
    max_outer_iterations=100,
    max_inner_iterations=10^4,
    acceleration_delay=10,
    projection=HomogeneousL0Projection,
)

iris_result = Dict()
iris_result["standard"] = Dict()
iris_result["projected"] = Dict()

for encoding in (:standard, :projected), algorithm in (MMSVD(), SD(), PGD())
    result = iris_benchmark(algorithm, encoding, 0.25, (1e-6, 1e-6, 1e-3), seed)
    iris_result[string(encoding)][string(typeof(algorithm))] = result
end

# #### Convergence History

plot_history(iris_result)

# #### Wall Time

plot_wall_time(iris_result)

# #### Prediction Accuracy

plot_accuracy(iris_result)

# ### Synthetic Dataset

seed = 1903
synthetic_benchmark = make_benchmark_runner("synthetic";
    lambda=1.0,
    max_outer_iterations=100,
    max_inner_iterations=10^4,
    acceleration_delay=10,
    projection=HomogeneousL0Projection,
)

synthetic_result = Dict()
synthetic_result["standard"] = Dict()
synthetic_result["projected"] = Dict()

for encoding in (:standard, :projected), algorithm in (MMSVD(), SD(), PGD())
    result = synthetic_benchmark(algorithm, encoding, 0.996, (1e-6, 1e-6, 1e-3), seed)
    synthetic_result[string(encoding)][string(typeof(algorithm))] = result
end

# #### Convergence History

plot_history(synthetic_result)

# #### Wall Time

plot_wall_time(synthetic_result)

# #### Prediction Accuracy

plot_accuracy(synthetic_result)

# `PGD()` looks like it struggles to converge to the correct support. Let's look at `accuracy` vs `gtol`.

synthetic_sensitivity_result = Dict()
synthetic_sensitivity_result["standard"] = Dict()
synthetic_sensitivity_result["projected"] = Dict()

gtols = [10.0 ^ -k for k in 0:8]
for encoding in (:standard, :projected), algorithm in (MMSVD(), SD(), PGD()), gtol in gtols
    result = synthetic_benchmark(algorithm, encoding, 0.996, (1e-6, 0.0, gtol), seed)
    key1, key2 = string(encoding), string(typeof(algorithm))
    if haskey(synthetic_sensitivity_result[key1], key2)
        xs, rs = synthetic_sensitivity_result[key1][key2]
    else
        xs, rs = Float64[], []
        synthetic_sensitivity_result[key1][key2] = (xs, rs)
    end
    push!(xs, gtol)
    push!(rs, result)
end

let
    colors = Makie.wong_colors()
    titles = ["Projected Simplex", "Standard Simplex"]
    labels = ["MMSVD", "SD", "PGD"]
    encodings = ["projected", "standard"]

    fig = Figure()
    ax = [Axis(fig[i,1];
            xscale=log10,
            title=titles[i],
            xlabel="log10[gtol]",
            ylabel="Accuracy",
            xminorticksvisible=true,
            yminorticksvisible=true,
            xminorticks=IntervalsBetween(9),
            yminorticks=IntervalsBetween(9),
        ) for i in 1:2
    ]

    for (i, enc) in enumerate(encodings)
        for (j, alg) in enumerate(labels)
            (xs, rs) = synthetic_sensitivity_result[enc][alg]
            ys = [r.accuracy for r in rs]
            scatterlines!(ax[i], xs, ys, label=alg, color=colors[j], linewidth=5,)
        end
    end

    elements = [PolyElement(polycolor=colors[j]) for j in eachindex(labels)]
    Legend(fig[1:2,2], elements, labels, "Algorithm")

    return fig
end

# Let's look at `loss` vs `gtol`.

let
    colors = Makie.wong_colors()
    titles = ["Projected Simplex", "Standard Simplex"]
    labels = ["MMSVD", "SD", "PGD"]
    encodings = ["projected", "standard"]

    fig = Figure()
    ax = [Axis(fig[i,1];
            xscale=log10,
            yscale=log10,
            title=titles[i],
            xlabel="log10[gtol]",
            ylabel="log10[loss]",
            xminorticksvisible=true,
            yminorticksvisible=true,
            xminorticks=IntervalsBetween(9),
            yminorticks=IntervalsBetween(9),
        ) for i in 1:2
    ]

    for (i, enc) in enumerate(encodings)
        for (j, alg) in enumerate(labels)
            (xs, rs) = synthetic_sensitivity_result[enc][alg]
            ys = [r.history[:loss][end] for r in rs]
            scatterlines!(ax[i], xs, ys, label=alg, color=colors[j], linewidth=5,)
        end
    end

    elements = [PolyElement(polycolor=colors[j]) for j in eachindex(labels)]
    Legend(fig[1:2,2], elements, labels, "Algorithm")

    return fig
end

# ### TCGA Dataset

seed = 1903
tcga_benchmark = make_benchmark_runner("TCGA-HiSeq";
    lambda=1.0,
    max_outer_iterations=100,
    max_inner_iterations=10^4,
    acceleration_delay=10,
    projection=HomogeneousL0Projection,
)

tcga_result = Dict()
tcga_result["standard"] = Dict()
tcga_result["projected"] = Dict()

for encoding in (:standard, :projected), algorithm in (MMSVD(), SD(), PGD())
    result = tcga_benchmark(algorithm, encoding, 0.99, (1e-6, 1e-6, 1e-3), seed)
    tcga_result[string(encoding)][string(typeof(algorithm))] = result
end

# #### Convergence History

plot_history(tcga_result)

# #### Wall Time

plot_wall_time(tcga_result)

# #### Prediction Accuracy

plot_accuracy(tcga_result)
