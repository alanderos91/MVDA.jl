#
#   Set Environment: examples/Project.toml + examples/Manifest.toml
#
import Pkg
Pkg.activate(".")

using MVDA
using MVDA.DataDeps
using CSV, DataFrames, DataFramesMeta, Latexify, LaTeXStrings, CairoMakie
using Statistics, StatsBase, LinearAlgebra
using Printf

const SETTINGS = Dict(
    :resolution => 72 .* (8.5*0.9, 11*0.3),
    :paper_size => (8.5, 11.0),
    :resolution_scaling_factor => 72,
    :fontsize => 8
)

const ABBREVIATIONS = Dict(
    "HomogeneousL0Projection" => "HomL0",
    "HeterogeneousL0Projection" => "HetL0",
    "HomogeneousL1BallProjection" => "HomL1",
    "HeterogeneousL1BallProjection" => "HetL1",
    "L1RSVM" => "L1R-SVM",
)

##### Helper functions #####
find_nz(x) = findall(x -> norm(x) != 0, eachrow(x))

function load_synth1_data(dir::AbstractString)
    data = CSV.read(joinpath(dir, "synth_homogeneous.csv"), DataFrame)
    transform!(data, [:true_features, :active_features] => ByRow((x, y) -> x / y) => :ppv)
    transform!(data, [:true_features, :k] => ByRow((x, y) -> x / y) => :tpr)
    transform!(data, [:accuracy_train, :accuracy_test] => ByRow((x, y) -> x - y) => :test_train_gap)
    return data
end

function load_synth2_data(dir::AbstractString)
    parse_vec(x) = [parse(Int, x.match) for x in eachmatch(r"\d+", x)]
    data = CSV.read(joinpath(dir, "synth_heterogeneous.csv"), DataFrame)
    for col in (:active_features, :true_features, :false_features)
        transform!(data, col => ByRow(parse_vec) => col)
    end
    transform!(data, [:true_features, :active_features] => ByRow((x, y) -> mean(replace(x ./ y, NaN => 0.0))) => :ppv)
    transform!(data, [:true_features, :k, :c] => ByRow((x, y, z) -> mean(x ./ (y ./ z))) => :tpr)
    transform!(data, [:accuracy_train, :accuracy_test] => ByRow((x, y) -> mean(x .- y)) => :test_train_gap)
    return data
end

function plot_cv_path(path_df, nvars; kwargs...)
    fig = Figure()
    ax = Axis(fig[1,1];
        width=800,
        height=400,
        xreversed=true,
        xticks = LinearTicks(min(16, nvars)),
        xlabel="Number of active variables",
        ylabel = "Classification Error",
        limits=(-1, nvars+1, 0, 100),
        yticks = LinearTicks(5),
        kwargs...
    )

    plot_cv_path!(ax, path_df, nvars)

    resize_to_layout!(fig)
    return fig
end

function plot_cv_path!(ax, path_df, nvars)
    path_gdf = groupby(path_df, [:title, :algorithm, :replicate, :k])
    score_df = combine(path_gdf, :validation => mean => identity)

    for df in groupby(score_df, :replicate)
        xs = df.k
        ys = 100 .* (1 .- df.validation)
        lines!(ax, xs, ys, color=(:black, 0.1), linewidth=3)
    end

    avg_path = combine(groupby(score_df, :k), :validation => mean => identity)
    xs = avg_path.k
    ys = 100 .* (1 .- avg_path.validation)
    lines!(ax, xs, ys, color=:red, linewidth=2, linestyle=:dash)

    return ax
end

function load_dfs(dir, example, proj)
  if proj == "L1RSVM"
    (
      nothing,
      nothing,
      CSV.read(joinpath(dir, example, proj, "modelA", "summary.out"), DataFrame),
      nothing,
    )
  else
    (
      CSV.read(joinpath(dir, example, proj, "cv_tune.out"), DataFrame),
      CSV.read(joinpath(dir, example, proj, "cv_path.out"), DataFrame),
      CSV.read(joinpath(dir, example, proj, "modelA", "summary.out"), DataFrame),
      CSV.read(joinpath(dir, example, proj, "modelB", "summary.out"), DataFrame)
    )
  end
end

function load_replicate_data(dir, example, proj, model, filename)
    example_dir = joinpath(dir, example, proj, model)
    dirs = readdir(example_dir, join=true)
    filter!(isdir, dirs)

    df = DataFrame()
    for (k, replicate_dir) in enumerate(dirs)
        tmp = CSV.read(joinpath(replicate_dir, filename), DataFrame)
        tmp.replicate .= k
        df = vcat(df, tmp)
    end

    return df
end

load_hist(dir, example, proj, model) = load_replicate_data(dir, example, proj, model, "history.out")
load_cmat(dir, example, proj, model) = load_replicate_data(dir, example, proj, model, "confusion_matrix.out")
load_prob(dir, example, proj, model) = load_replicate_data(dir, example, proj, model, "probability_matrix.out")

function load_coefficients(dir, example, projection, model)
    example_dir = joinpath(dir, example, projection, model)
    dirs = readdir(example_dir, join=true)
    filter!(isdir, dirs)

    arr = []
    for (k, replicate_dir) in enumerate(dirs)
        push!(arr, MVDA.load_model(replicate_dir))
    end
    
    return arr
end

function plot_convergence_data!(ax, df, metric)
    ys = log10.(replace(df[!, metric], 0.0 => 1.0))
    lines!(ax, ys, color=(:black, 0.2), linewidth=3)
    return ax
end

function plot_convergence_data(df)
    fig = Figure()
    ax = [Axis(fig[i,1], yticks=1:-1:-6) for i in 1:5]

    ax[1].ylabel = "risk"
    ax[2].ylabel = "loss"
    ax[3].ylabel = "objective"
    ax[4].ylabel = "distance"
    ax[5].ylabel = "gradient"

    for rep in groupby(df, :replicate)
        plot_convergence_data!(ax[1], rep, :risk)
        plot_convergence_data!(ax[2], rep, :loss)
        plot_convergence_data!(ax[3], rep, :objective)
        plot_convergence_data!(ax[4], rep, :distance)
        plot_convergence_data!(ax[5], rep, :gradient)
    end

    resize_to_layout!(fig)

    return fig
end

function save_table(filepath, df::DataFrame, ::Nothing)
    table_str = latexify(df,
        latex=false,
        env=:table,
        booktabs=true,
        adjustment=:l
    )
    write(filepath, table_str)
end

function save_table(filepath, df::DataFrame, header)
    table_str = latexify(df,
        latex=false,
        env=:table,
        booktabs=true,
        head=header,
        adjustment=:l
    )
    write(filepath, table_str)
end

function save_table(filepath, table_str::String, ::Any)
    write(filepath, table_str)
end

##### Figures #####

function cancer_size_distributions(dir, projection)
    examples = [
        "colon" "srbctA" "leukemiaA"
        "lymphomaA" "brain" "prostate"
    ]

    titles = [
        "Colon (p = 2000)" "SRBCT (p = 2308)" "Leukemia (p = 3571)"
        "Lymphoma (p = 4026)" "Brain (p = 5597)" "Prostate (p = 6033)"
    ]

    global SETTINGS
    fz = SETTINGS[:fontsize]
    fig = Figure(size=(SETTINGS[:resolution][1], SETTINGS[:resolution][2]*1.25), fontsize=SETTINGS[:fontsize])
    g = fig[1,1] = GridLayout(alignmode=Outside(5))
    common_options = (;
        titlesize=1.25*fz,
        xticklabelsize=fz,
        yticklabelsize=fz,
        xticks=LinearTicks(9),
        xticklabelrotation=pi/6,
        xlabelpadding=0,
        ylabelpadding=0,
    )

    Label(g[3, 2:4], "Number of active features", fontsize=1.25*fz, tellwidth=false)
    Label(g[1:2, 1], "Frequency", rotation=pi/2, fontsize=1.25*fz, tellheight=false)
    ax = [Axis(g[i,j+1]; title=titles[i,j], common_options...) for i in 1:2, j in 1:3]
    linkyaxes!(ax...)

    for j in 1:3, i in 1:2
        _, _, modelA, _ = load_dfs(dir, examples[i,j], projection)
        hist!(ax[i,j], modelA.active_variables, bins=32)
    end

    colgap!(g, 5)
    rowgap!(g, 5)
    resize_to_layout!(fig)

    return fig
end

function TCGA_topgenes(dir, projection)
    global SETTINGS
    fz = SETTINGS[:fontsize]

    df = MVDA.dataset("TCGA-HiSeq")
    L, X = Vector{String}(df[!,1]), Matrix{Float64}(df[!,2:end])
    problem = MVDAProblem(L, X, intercept=false, encoding=:standard)
    class_reference = problem.labels
    n_classes = length(class_reference)

    reffile = joinpath(dirname(dir), "TCGA-PANCAN-HiSeq-genes.csv")
    gene_reference = CSV.read(reffile, DataFrame)[!,:symbol]
    selected_genes = CSV.read(datadep"MVDA/TCGA-HiSeq.cols", DataFrame, header=false)[1:end-2,1]
    genes = gene_reference[selected_genes]
    common_kwargs = (;
        titlesize=1.25*fz,
        xticklabelsize=fz,
        xticklabelrotation=pi/4,
        yticks=LinearTicks(6),
    )

    coeff = [MVDA.load_model(joinpath(dir, "TCGA-HiSeq", projection, "modelA", "$(k)")).coeff_proj.slope for k in 1:10]

    # compute counts and average coefficients
    gene_count = zeros(Int, size(coeff[1]))
    for i in axes(gene_count, 2)
        eff = map(x -> vec(x[:, i]), coeff)
        nz_row = map(find_nz, eff)
        for arr in nz_row, j in arr
            gene_count[j, i] += 1
        end
    end
    nz_counts = copy(gene_count)
    replace!(nz_counts, 0 => 1)
    scaled_avg_effect_size = (1 ./ nz_counts) .* sum(coeff)

    # init plot
    resolution, factor = SETTINGS[:paper_size], SETTINGS[:resolution_scaling_factor]
    resolution = (factor * 0.95 * resolution[1], factor * 0.17 * n_classes * resolution[2])
    coeff_abs_max = 1e-1
    crange = [-coeff_abs_max, coeff_abs_max] * 1e4
    cscale = Makie.Symlog10(-10, 10)
    cmap = cgrad(:RdBu, 1001, scale=cscale)

    fig = Figure(size=resolution, fontsize=fz)
    g = GridLayout(fig[1,1])
    ax = [Axis(g[j,1]; common_kwargs...) for j in 1:n_classes]
    for i in eachindex(ax)
        coeff_i = scaled_avg_effect_size[:, i] * 1e4
        count_i = gene_count[:, i]
        gene_key = copy(genes)

        # Rank the genes from most replicated to least replicated
        idx = sortperm(count_i, rev=true)
        gene_key, count_i, coeff_i = gene_key[idx], count_i[idx], coeff_i[idx]
        topK = min(length(find_nz(count_i)), 50)

        xlims!(ax[i], low=0, high=topK+1)
        barplot!(ax[i], count_i[1:topK];
            color=coeff_i[1:topK],
            colormap=cmap,
            colorrange=crange,
            colorscale=cscale,
            direction=:y,
        )
        ax[i].title = "$(class_reference[i])"       # TCGA cancer
        ax[i].xticks = (1:topK, gene_key[1:topK])   # label genes
    end

    # Set common x-label
    Label(g[2:end-1, 0], "Frequency", fontsize=1.25*fz, rotation=pi/2)

    # Set common colorbar
    Colorbar(g[2:end-1, 2];
        label="Slope Coefficient × 10⁴",
        labelsize=1.25*fz,
        colormap=cmap,
        colorrange=crange,
        scale=cscale,
        # vertical=false,
        # flipaxis=false,
        labelpadding=0,
        ticks=[-1000, -100, -10, 0, 10, 100, 1000],
    )

    colgap!(g, 1, 5)
    rowgap!(g, 0)

    return fig
end

function boxplot_summary(df, metric, metric_label,
    yticks=LinearTicks(11), ymin=nothing, ymax=nothing;
    size=(800, 600)
)
    #
    global SETTINGS
    fz = SETTINGS[:fontsize]

    ns = sort!(unique(df.n))
    cs = sort!(unique(df.c))
    snrs = sort!(unique(df.SNR))
    
    palette = Makie.wong_colors()
    solvers = unique(df.solver)
    solver_bin = Dict(solver => k for (k, solver) in enumerate(solvers))
    subplot_label = string.('A':'Z')

    fig = Figure(size=size, fontsize=fz)
    grd = GridLayout(fig[1,1])
    axs = []
    for j in eachindex(cs), i in eachindex(ns)
        n = ns[i]
        c = cs[j]
        
        srd = GridLayout(grd[i, j], alignmode=Outside(5))
        box = Box(grd[i, j], cornerradius=0, color=:transparent, strokecolor=:lightgray)
        Makie.translate!(box.blockscene, 0, 0, -100)

        Label(srd[1, 1], subplot_label[length(ns)*(j-1)+i],
            halign = :left,
            valign = :top,
            font = :bold,
            fontsize = 1.25*fz,
            padding = (3, 0, 0, 0),
            tellwidth=false,
            tellheight=false,
        )
        Label(srd[1, 2], "n = $(n), c = $(c)";
            halign = :center,
            valign = :bottom,
            font = :bold,
            fontsize = 1.25fz,
            padding = (0, 0, 0, 0),
            tellwidth=false,
        )

        if isnothing(ymin) && isnothing(ymax)
            _ymin, _ymax = extrema(@rsubset(df, :n == n, :p == 1000, :c == c, :k == 30)[!, metric])
            _ymin, _ymax = _ymin*0.95, _ymax*1.05
        else
            _ymin, _ymax = ymin, ymax
        end

        for k in eachindex(snrs)
            snr = snrs[k]

            data_subset = @rsubset(df, :n == n, :p == 1000, :c == c, :k == 30, :SNR == snr)
            sort!(data_subset, :rho, lt=isless, rev=false)
            rhos = sort!(unique(data_subset.rho))

            # itr = Iterators.product(SNRs, rhos) # SNRs increases first
            # scenarios = [(rho, SNR) => idx for (idx, (SNR, rho)) in enumerate(itr)] |> vec
            scenario_bin = Dict(rho => idx for (idx, rho) in enumerate(rhos))
            xticklabels = ["ρ = $(rho)" for rho in rhos]

            categories, dodge, colors = Int[], Int[], []
            for row in eachrow(data_subset)
                scenario_idx = scenario_bin[row.rho]
                solver_idx = solver_bin[row.solver]
                push!(categories, scenario_idx)
                push!(dodge, solver_idx)
                push!(colors, palette[solver_idx])
            end

            # Label(srd[2, k], "SNR = $(snr)";
            Label(srd[k+1, 1], "SNR = $(snr)";
                # valign = :bottom,
                fontsize = 1.2*fz,
                font = :bold,
                padding=(0.5, 0.5, 0.5, 0.5),
                # tellwidth=false,
                # tellheight=false,
                rotation=pi/2,
            )

            xticks = (eachindex(xticklabels), xticklabels)
            # ax = Axis(srd[3, k];
            ax = Axis(srd[k+1, 2];
                xticks=xticks,
                xticklabelsvisible=(k == length(snrs)),
                # xticks=yticks,
                xticklabelsize=1.1*fz,
                xlabelpadding=0,
                # xticklabelrotation=pi/8,
                # ylabel=metric_label,
                ylabelsize=fz,
                yticks=yticks,
                # yticks=(eachindex(xticklabels), xticklabels),
                yticklabelsize=fz,
                ylabelpadding=0,
                ygridvisible=true,
                # yreversed=true,
                limits=(0.5, length(rhos)+0.5, _ymin, _ymax),
                # limits=(_ymin, _ymax, 0.5, length(rhos)+0.5),
            )
            push!(axs, ax)
            boxplot!(ax, categories, data_subset[!, metric];
                label=Vector(data_subset[!, :solver]),
                color=colors,
                dodge=dodge,
                # orientation=:horizontal,
            )
        end
        colgap!(srd, 10)
        rowgap!(srd, 10)
        rowgap!(srd, 1, 2)
    end

    Label(grd[0, 1:length(cs)], metric_label;
        # rotation=pi/2,
        fontsize=1.5*fz,
        font=:bold,
        # tellheight=false,
    )
    solvers = unique(df.solver)
    Legend(grd[length(ns)+1, 1:length(cs)],
        [PolyElement(color=palette[solver_bin[s]], strokecolor=:transparent) for s in solvers],
        solvers,
        "Methods";
        titlegap=0,
        patchsize=(30, 5),
        titlesize=1.25*fz,
        labelsize=1.1*fz,
        labelfont=:bold,
        framevisible=false,
        orientation=:horizontal
    )

    colgap!(grd, 10)
    rowgap!(grd, 8)
    resize_to_layout!(fig)

    return fig, grd, axs
end

function draw_time_boxplot(data)
    global SETTINGS
    C, sz = SETTINGS[:resolution_scaling_factor], SETTINGS[:paper_size]
    fig, grd, axs = with_theme(theme_minimal(), figure_padding=5) do
        boxplot_summary(
            data,
            :time, "Time [seconds]", LinearTicks(9);
            size = (0.9 * C * sz[1], 0.5 * C * sz[2]),
        )
    end
    return fig
end

function draw_train_acc_boxplot(data)
    global SETTINGS
    C, sz = SETTINGS[:resolution_scaling_factor], SETTINGS[:paper_size]
    fig, grd, axs = with_theme(theme_minimal(), figure_padding=5) do
        boxplot_summary(
            data,
            :accuracy_train, "Training Accuracy", LinearTicks(11), -0.05, 1.05;
            size = (0.9 * C * sz[1], 0.5 * C * sz[2]),
        )
    end
    return fig
end

function draw_test_acc_boxplot(data)
    global SETTINGS
    C, sz = SETTINGS[:resolution_scaling_factor], SETTINGS[:paper_size]
    fig, grd, axs = with_theme(theme_minimal(), figure_padding=5) do
        boxplot_summary(
            data,
            :accuracy_test, "Test Accuracy", LinearTicks(11), -0.05, 1.05;
            size = (0.9 * C * sz[1], 0.5 * C * sz[2]),
        )
    end
    return fig
end

function draw_gap_boxplot(data)
    global SETTINGS
    C, sz = SETTINGS[:resolution_scaling_factor], SETTINGS[:paper_size]
    fig, grd, axs = with_theme(theme_minimal(), figure_padding=5) do
        boxplot_summary(
            data,
            :test_train_gap, L"\text{Train} - \text{Test}", LinearTicks(11), -0.05, 1.05;
            size = (0.9 * C * sz[1], 0.5 * C * sz[2]),
        )
    end
    return fig
end

function draw_ppv_boxplot(data)
    global SETTINGS
    C, sz = SETTINGS[:resolution_scaling_factor], SETTINGS[:paper_size]
    fig, grd, axs = with_theme(theme_minimal(), figure_padding=5) do
        boxplot_summary(
            data,
            :ppv, "Ratio of True Positives to Predicted Positives", LinearTicks(7), -0.05, 1.05;
            size = (0.9 * C * sz[1], 0.5 * C * sz[2]),
        )
    end
    return fig
end

function draw_tpr_boxplot(data)
    global SETTINGS
    C, sz = SETTINGS[:resolution_scaling_factor], SETTINGS[:paper_size]
    fig, grd, axs = with_theme(theme_minimal(), figure_padding=5) do
        boxplot_summary(
            data,
            :tpr, "True Positive Rate", LinearTicks(7), -0.05, 1.05;
            size = (0.9 * C * sz[1], 0.5 * C * sz[2]),
        )
    end
    return fig
end

##### Tables #####

function timing_table(dir, examples, projs)
    #
    function gather_time_data(df)
        return combine(groupby(df, :replicate), :time => sum)[!, 2]
    end
    global ABBREVIATIONS
    colnames = String[""; map(id -> ABBREVIATIONS[id], projs)]
    df = rename!(DataFrame(Matrix{String}(undef, 0, length(colnames)), :auto), colnames)
    for example in examples
        newrow = String[]
        push!(newrow, example)
        for proj in projs
            tbls = load_dfs(dir, example, proj)
            ts = mapreduce(gather_time_data, +, tbls)
            push!(newrow, str_summarize_col(ts, 0.1, number_formatter2))
        end
        push!(df, newrow)
    end
    return df
end

function summarize_col(xs, alpha, f)
    lo = alpha
    hi = 1-lo
    l, v, h = map(f, (quantile(xs, lo), median(xs), quantile(xs, hi)))
    l, h = extrema((l, h))
    return (lo=l, val=v, hi=h)
end

summarize_col(df, col, alpha, f) = summarize_col(df[!, col], alpha, f)

error_percent_formatter(x) = round(100*(1-x), sigdigits=2)
number_formatter(x) = round(x, sigdigits=2)
number_formatter2(x) = round(x, sigdigits=3)
integer_formatter(x) = round(Int, x)

function str_summarize_col(args...)
    x = summarize_col(args...)
    @sprintf("%g, %g, %g", x.lo, x.val, x.hi)
end

function fetch_metric_label(col)
    LABELS = Dict(
        :train => "Train Error (\\%)",
        :test => "Test Error (\\%)",
        :epsilon => "\$\\epsilon\$",
        :hyperparam => "\$k\$ or \$\\lambda\$",
        :active_variables => "\\# Active",
        :time => "Time (s)",
    )

    return LABELS[col]
end

function fetch_metric_formatter(col)
    FORMATTER = Dict(
        :train => error_percent_formatter,
        :test => error_percent_formatter,
        :epsilon => number_formatter2,
        :lambda => number_formatter2,
        :k => integer_formatter,
        :active_variables => integer_formatter,
        :time => number_formatter,
    )
    return FORMATTER[col]
end

function fetch_metric((df1, df2), col, alpha)
    if col == :time
        data = combine(groupby(df1, :replicate), :time => sum => :time)
        data.time .+= df2.time
    else
        data = df2
    end
    fmt = fetch_metric_formatter(col)
    str_summarize_col(data, col, alpha, fmt)
end

function make_comparative_table_by_example(dir, examples, projections; alpha=0.05)
    global ABBREVIATIONS

    io = IOBuffer()

    ncolumns = length(projections)
    nsubcols = 2
    write(io, "\\begin{tabular}{c$(repeat("c", ncolumns*nsubcols))}\n")
    write(io, "\\toprule\n")
    col_label, col_rule = String[], String[]
    for (k, projection) in enumerate(projections)
      st = 2+nsubcols*(k-1)
      sp = st+nsubcols-1
      push!(col_label, "& \\multicolumn{$(nsubcols)}{c}{$(ABBREVIATIONS[projection])}")
      push!(col_rule,  "    \\cmidrule(lr){$(st)-$(sp)}")
    end
    write(io, "    $(join(col_label, "")) \\\\\n")
    write(io, "$(join(col_rule, ""))\n")
    write(io, "    $(repeat("& Error (\\%) & \\# Active ", ncolumns))\\\\\n")
    write(io, "\\midrule\n")

    for example in examples
        write(io, "$(example)")
        for projection in projections
          if ispath(joinpath(dir, example, projection))
            _, unused, df, _ = load_dfs(dir, example, projection)
            data_err = fetch_metric((unused, df), :test, alpha)
            data_act = fetch_metric((unused, df), :active_variables, alpha)
          else
            data_err = "Omitted"
            data_act = "Omitted"
          end
          write(io, " & $(data_err) & $(data_act)")
        end
        write(io, " \\\\\n")
    end

    write(io, "\\bottomrule\n")
    write(io, "\\end{tabular}")
    tbl = String(take!(io))
    close(io)
    return tbl
end

function synthetic_table_summary(df)
    # do not select p or k because they are assumed to be fixed
    params = [:n, :c, :rho, :SNR]
    sort!(df, [:n, :c, :SNR, :rho])
    solvers = unique(df.solver)
    gdf = groupby(df, [params; :solver])

    M = length(params)
    N = length(solvers)

    # time
    tmp = unstack(
        combine(gdf, :time => (x -> str_summarize_col(x, 0.1, number_formatter2)) => :time),
        :solver,
        :time;
        renamecols = x -> Symbol(:time_, x)
    )

    # test error
    tmp2 = unstack(
        combine(gdf, :accuracy_test => (x -> str_summarize_col(x, 0.1, error_percent_formatter)) => :test),
        :solver,
        :test;
        renamecols = x -> Symbol(:test_, x)
    )
    tmp = leftjoin(tmp, tmp2; on=params)

    io = IOBuffer()

    write(io, "\\begin{tabular}{$(repeat('c', M+2*N))}\n")
    write(io, "\\toprule\n")
    write(io, "    $(repeat(" & ", M))\\multicolumn{$(N)}{c}{Time (s)} & \\multicolumn{$(N)}{c}{Test (\\%)} \\\\\n")
    write(io, "    \\cmidrule(lr){$(N+1)-$(N+M)}    \\cmidrule(lr){$(N+M+1)-$(N+2*M)} \n")
    write(io, "    $(join(params, " & ")) & $(join(solvers, " & ")) & $(join(solvers, " & ")) \\\\\n")
    write(io, "\\midrule\n")

    for row in eachrow(tmp)
        write(io, join(row, " & "))
        write(io, " \\\\\n")
    end

    write(io, "\\bottomrule\n")
    write(io, "\\end{tabular}")
    tbl = String(take!(io))
    close(io)
    return tbl
end

function detailed_results(dir, example, projections, nreps; alpha=0.05)
  prob(x...) = x ./ sum(x)
  global ABBREVIATIONS

  # Collect results for each scenario
  classes = String[]
  r = Dict()
  for projection in projections
    # Overall
    results = Dict(); results["total"] = Dict()
    _, unused, df, _ = load_dfs(dir, example, projection)

    # Load confusion matrix and coefficients.
    tmpcfmat = [CSV.read(joinpath(dir, example, projection, "modelA", "$(k)", "confusion_matrix.out"), DataFrame) for k in 1:nreps]
    if projection == "L1RSVM"
      coeff = [zeros(1, 1) for k in 1:nreps]
    else
      coeff = [MVDA.load_model(joinpath(dir, example, projection, "modelA", "$(k)")).coeff_proj.slope for k in 1:nreps]
    end
    
    # Count the number of times each gene is selected in CV replicates.
    gene_count = zeros(Int, size(coeff[1]))
    for i in axes(gene_count, 2)
        eff = map(x -> vec(x[:, i]), coeff)
        nz_row = map(find_nz, eff)
        for arr in nz_row, j in arr
          gene_count[j, i] += 1
        end
    end

    # overall
    results["total"]["error"] = fetch_metric((unused, df), :test, alpha)
    results["total"]["active"] = fetch_metric((unused, df), :active_variables, alpha)

    # Extract class labels.
    if isempty(classes)
      for class in names(tmpcfmat[1])[3:end]
        push!(classes, class)
        results[class] = Dict()
      end
    else
      for class in classes
        results[class] = Dict()
      end
    end

    # Reformat the confusion matrix into a true DataFrame.
    cfmat = DataFrame()
    for (k, df) in enumerate(tmpcfmat)
      filter!("subset" => isequal("test"), df)
      select!(df, Not(1:2))
      transform!(df, names(df) => ByRow(prob) => names(df); renamecols=false)
      tmp = Matrix(df) |> diag |> transpose |> Tables.table |> DataFrame
      rename!(tmp, names(df))
      tmp[!, "replicate"] .= k
      cfmat = vcat(cfmat, tmp)
    end

    # Class-specific
    for (k, class) in enumerate(classes)
      results[class]["error"] = str_summarize_col(cfmat, class, 0.1, error_percent_formatter)
      if projection == "L1RSVM"
        results[class]["active"] = ""
      else
        G = maximum(gene_count)
        most_stable = count(==(G), gene_count[:, k])
        results[class]["active"] = string(integer_formatter(G), ", ", integer_formatter(most_stable))
      end
    end

    r[projection] = results
  end

  io = IOBuffer()
  ncolumns = length(projections)
  nsubcols = 2
  write(io, "\\begin{tabular}{c$(repeat("c", ncolumns*nsubcols))}\n")
  write(io, "\\toprule\n")
  col_label, col_rule = String[], String[]
  for (k, projection) in enumerate(projections)
    st = 2+nsubcols*(k-1)
    sp = st+nsubcols-1
    push!(col_label, "& \\multicolumn{$(nsubcols)}{c}{$(ABBREVIATIONS[projection])}")
    push!(col_rule,  "    \\cmidrule(lr){$(st)-$(sp)}")
  end
  write(io, "    $(join(col_label, "")) \\\\\n")
  write(io, "$(join(col_rule, ""))\n")
  write(io, "    $(repeat("& Error (\\%) & \\# Active ", ncolumns))\\\\\n")
  write(io, "\\midrule\n")

  # Add overall results
  write(io, "Total ")
  for key in projections
    write(io, """
        & $(r[key]["total"]["error"])
        & $(r[key]["total"]["active"])
    """)
  end
  write(io, "\\\\\n")

  # Add class-specific results
  for class in classes
    write(io, "$(class) ")
    for key in projections
      write(io, """
          & $(r[key][class]["error"])
          & $(r[key][class]["active"])
      """)
    end
    write(io, "\\\\\n")
  end

  write(io, "\\bottomrule\n")
  write(io, "\\end{tabular}")
  tbl = String(take!(io))
  close(io)
  return tbl
end

function pam50_overlap(dir, projections, nreps)
  global ABBREVIATIONS

  # Gather gene names
  reffile = joinpath(dirname(dir), "TCGA-BRCA-preprocessed_genes.csv")
  genes = CSV.read(reffile, DataFrame)[!,:name]
  pam50 = CSV.read(joinpath(dirname(dir), "PAM50.txt"), DataFrame; header=false)[!, 1]

  df = DataFrame(method=String[], Basal=[], Her2=[], LumA=[], LumB=[], Normal=[])
  for projection in projections
    coeff = [MVDA.load_model(joinpath(dir, "BRCA", projection, "modelA", "$(k)")).coeff_proj.slope for k in 1:nreps]    
    gene_count = zeros(Int, size(coeff[1]))
    for i in axes(gene_count, 2)
        eff = map(x -> vec(x[:, i]), coeff)
        nz_row = map(find_nz, eff)
        for arr in nz_row, j in arr
          gene_count[j, i] += 1
        end
    end
    x = String[]
    for k in axes(gene_count, 2)
      idx = findall(==(maximum(gene_count)), gene_count[:, k])     # most replicated genes
      overlap = intersect(pam50, genes[idx])
      if 1 <= length(overlap) <= 25
        push!(x, "(+) " * join(overlap, ","))
      elseif length(overlap) > 25
        push!(x, "(-): " * join(setdiff(pam50, overlap), ","))
      else
        push!(x, "")
      end
    end
    push!(df, (ABBREVIATIONS[projection], x...))
  end
  return df
end

##### Execution of the script #####

function main(input, output)
    synth_path = joinpath(input, "synthetic")
    uci_path = joinpath(input, "linear")
    cancer_path = joinpath(input, "cancer")
    nonlinear_path = joinpath(input, "nonlinear")

    figures = joinpath(output, "figures")
    tables = joinpath(output, "tables")
    for dir in (figures, tables)
        if !ispath(dir)
            mkpath(dir)
        end
    end

    # Section 4.1
    @info "Generating Table 2..."
    @time begin
        synth1 = load_synth1_data(synth_path)
        tbl2 = synthetic_table_summary(synth1)
        save_table(joinpath(tables, "Table2.tex"), tbl2, nothing)
    end

    @info "Generating Table 3..."
    @time begin
        synth2 = load_synth2_data(synth_path)
        tbl3 = synthetic_table_summary(synth2)
        save_table(joinpath(tables, "Table3.tex"), tbl3, nothing)
    end

    @info "Generating Figure 2..."
    @time begin
        fig2 = draw_tpr_boxplot(synth1)
        save(joinpath(figures, "Figure2.pdf"), fig2, pt_per_unit=1)
    end

    @info "Generating Figure 3..."
    @time begin
        fig3 = draw_ppv_boxplot(synth1)
        save(joinpath(figures, "Figure3.pdf"), fig3, pt_per_unit=1)
    end

    @info "Generating Figure 4..."
    @time begin
        fig4 = draw_tpr_boxplot(synth2)
        save(joinpath(figures, "Figure4.pdf"), fig4, pt_per_unit=1)
    end

    @info "Generating Figure 5..."
    @time begin
        fig5 = draw_ppv_boxplot(synth2)
        save(joinpath(figures, "Figure5.pdf"), fig5, pt_per_unit=1)
    end

    # Section 4.2
    @info "Generating Table 4..."
    @time begin
        tbl4 = make_comparative_table_by_example(cancer_path,
            [
                "colon", "srbctA", "leukemiaA", "lymphomaA", "brain",
                "prostate",
            ],
            [
                "HomogeneousL0Projection",
                "HeterogeneousL0Projection",
                "HeterogeneousL1BallProjection",
            ];
            alpha=0.1
        )
        save_table(joinpath(tables, "Table4.tex"), tbl4, nothing)
    end
    
    @info "Generating Figure 6..."
    @time begin
        fig6 = with_theme(theme_minimal(), figure_padding=5) do
            cancer_size_distributions(cancer_path, "HeterogeneousL0Projection")
        end
        save(joinpath(figures, "Figure6.pdf"), fig6, pt_per_unit=1)
    end

    # Section 4.3
    @info "Generating Table 5..."
    @time begin
        tbl5 = make_comparative_table_by_example(uci_path,
            [
                "iris", "lymphography", "zoo", "bcw","waveform",
                "splice", "letters", "optdigits", "vowel", "HAR",
                "TCGA-HiSeq",
            ],
            [
                "HomogeneousL0Projection",
                "HeterogeneousL0Projection",
                "HeterogeneousL1BallProjection",
                "L1RSVM",
            ];
            alpha=0.1
        )
        save_table(joinpath(tables, "Table5.tex"), tbl5, nothing)
    end

    @info "Generating Figure 7..."
    @time begin
        fig7 = with_theme(theme_minimal(), figure_padding=5) do
            TCGA_topgenes(uci_path, "HeterogeneousL0Projection")
        end
        save(joinpath(figures, "Figure7.pdf"), fig7, pt_per_unit=1)
    end

    # Section 4.4
    @info "Generating Table 6..."
    @time begin
        tbl6 = make_comparative_table_by_example(nonlinear_path,
            [
                "circles", "clouds", "waveform", "spiral", "spiral-hard",
                "vowel",
            ],
            [
                "HomogeneousL0Projection",
                "HeterogeneousL0Projection",
                "HeterogeneousL1BallProjection",
            ];
            alpha=0.1
        )
        save_table(joinpath(tables, "Table6.tex"), tbl6, nothing)
    end

    @info "Generating Table 7..."
    @time begin
        tbl7 = detailed_results(uci_path, "BRCA",
            [
                "HomogeneousL0Projection",
                "HeterogeneousL0Projection",
                "HeterogeneousL1BallProjection",
                "L1RSVM",
            ],
            10; # number of replicates
            alpha=0.1
        )
        save_table(joinpath(tables, "Table7.tex"), tbl7, nothing)
    end

    @info "Generating Table 8..."
    @time begin
        tbl8 = detailed_results(uci_path, "TGP",
            [
                "HomogeneousL0Projection",
                "HeterogeneousL0Projection",
                "HeterogeneousL1BallProjection",
            ],
            10; # number of replicates
            alpha=0.1
        )
        save_table(joinpath(tables, "Table8.tex"), tbl8, nothing)
    end

    # Appendix: Timing
    @info "Generating Appendix Table C1"
    @time begin
        tblC1 = timing_table(cancer_path,
            [
                "colon", "srbctA", "leukemiaA", "lymphomaA", "brain",
                "prostate",
            ],
            [
                "HomogeneousL0Projection",
                "HeterogeneousL0Projection",
                "HeterogeneousL1BallProjection",
            ]
        )
        save_table(joinpath(tables, "TableC1.tex"), tblC1, nothing)
    end

    @info "Generating Appendix Table C2"
    @time begin
        tblC2 = timing_table(uci_path,
            [
                "iris", "lymphography", "zoo", "bcw","waveform",
                "splice", "letters", "optdigits", "vowel", "HAR",
                "TCGA-HiSeq", "BRCA", "TGP",
            ],
            [
                "HomogeneousL0Projection",
                "HeterogeneousL0Projection",
                "HeterogeneousL1BallProjection",
            ]
        )
        save_table(joinpath(tables, "TableC2.tex"), tblC2, nothing)
    end

    @info "Generating Appendix Table C3"
    @time begin
        tblC3 = pam50_overlap(uci_path,
          [
            "HomogeneousL0Projection",
            "HeterogeneousL0Projection",
            "HeterogeneousL1BallProjection",
          ],
          10 # number of replicates
        )
        save_table(joinpath(tables, "TableC3.tex"), tblC3, nothing)
    end

    return nothing
end

# Run the script.
input_dir, output_dir = ARGS[1], ARGS[2]
main(input_dir, output_dir)
