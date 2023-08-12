using MVDA, CairoMakie, CSV, DataFrames, Statistics, StatsBase, Latexify, LaTeXStrings, LinearAlgebra

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
)

##### Helper functions #####
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

load_dfs(dir, example, proj) = (
    CSV.read(joinpath(dir, example, proj, "cv_path.out"), DataFrame),
    CSV.read(joinpath(dir, example, proj, "modelA", "summary.out"), DataFrame),
    CSV.read(joinpath(dir, example, proj, "modelB", "summary.out"), DataFrame)
)

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

function synthetic_coefficients(dir)
    mse(x, y) = mean( (x .- y) .^ 2 )

    arrA = load_coefficients(dir, "synthetic", "HomogeneousL0Projection", "modelA")
    arrB = load_coefficients(dir, "synthetic", "HomogeneousL0Projection", "modelB")

    modelA = (; x1=Float64[], x2=Float64[])
    modelB = (; x1=Float64[], x2=Float64[])
    for (A, B) in zip(arrA, arrB)
        push!(modelA.x1, A.coeff_proj.slope[1])
        push!(modelA.x2, A.coeff_proj.slope[2])
        push!(modelB.x1, B.coeff.slope[1])
        push!(modelB.x2, B.coeff.slope[2])
    end

    global SETTINGS
    fig = Figure(resolution=SETTINGS[:resolution], fontsize=SETTINGS[:fontsize])
    g = fig[1,1] = GridLayout()

    Box(g[1,2], color=:gray90, tellwidth=false)
    Label(g[1,2], "Sparse Model", tellwidth=false)

    Box(g[1,3], color=:gray90, tellwidth=false)
    Label(g[1,3], "Reduced Model", tellwidth=false)
    
    Box(g[1,4], color=:gray90, tellwidth=false)
    Label(g[1,4], "Difference", tellwidth=false)

    Box(g[2,1], color=:gray90, tellheight=false)
    Label(g[2,1], "Feature 1", rotation=pi/2, tellheight=false)
    
    Box(g[3,1], color=:gray90, tellheight=false)
    Label(g[3,1], "Feature 2", rotation=pi/2, tellheight=false)

    ax = [Axis(g[i+1,j+1]) for i in 1:2, j in 1:3]

    hist!(ax[1,1], modelA.x1)
    hist!(ax[1,2], modelB.x1)
    hist!(ax[1,3], modelA.x1 .- modelB.x1)
    hist!(ax[2,1], modelA.x2)
    hist!(ax[2,2], modelB.x2)
    hist!(ax[2,3], modelA.x2 .- modelB.x2)

    Label(g[4,2:4], "Estimated Value")

    colgap!(g, 5)
    rowgap!(g, 5)

    return fig
end

function synthetic_summary(dir)
    path_df1, _, _ = load_dfs(dir, "synthetic", "HomogeneousL0Projection")
    path_df2, _, _ = load_dfs(dir, "synthetic-hard", "HomogeneousL0Projection")
    
    hist1 = load_hist(dir, "synthetic", "HomogeneousL0Projection", "modelA")
    hist2 = load_hist(dir, "synthetic-hard", "HomogeneousL0Projection", "modelA")

    nvars = 500
    kwargs = (;
        xreversed=true,
        xticks = LinearTicks(min(11, nvars)),
        limits=(-1, nvars+1, 0, 100),
        yticks = LinearTicks(5),
    )

    global SETTINGS
    fig = Figure(resolution=SETTINGS[:resolution], fontsize=SETTINGS[:fontsize])
    g = fig[1,1] = GridLayout()

    Box(g[1,2:3], color=:gray90, tellheight=false)
    Label(g[1,2:3], "Classification error (%)", tellwidth=false)
    
    Box(g[1,4], color=:gray90, tellheight=false)
    Label(g[1,4], "log[Gradient norm]", tellwidth=false)

    Box(g[1,5], color=:gray90, tellheight=false)
    Label(g[1,5], "log[Objective]", tellwidth=false)

    Box(g[2,1], color=:gray90, tellheight=false)
    Label(g[2,1], "synthetic", rotation=pi/2, tellheight=false)
    
    Box(g[3,1], color=:gray90, tellheight=false)
    Label(g[3,1], "synthetic-hard", rotation=pi/2, tellheight=false)

    Label(g[4,2:3], "Number of active variables", tellwidth=false)
    Label(g[4,4:5], "Iteration", tellwidth=false)

    left_panels = [Axis(g[i+1,2:3]; kwargs...) for i in 1:2]
    right_panels = [Axis(g[i+1,j+1], xticks=LinearTicks(5)) for i in 1:2, j in 3:4]
    ax = [ left_panels right_panels ]

    linkxaxes!(ax[1,2], ax[1,3])
    linkxaxes!(ax[2,2], ax[2,3])

    linkyaxes!(ax[1,2], ax[2,2])
    linkyaxes!(ax[1,3], ax[2,3])

    plot_cv_path!(ax[1,1], path_df1, nvars)
    plot_cv_path!(ax[2,1], path_df2, nvars)

    for df in groupby(hist1, :replicate)
        lines!(ax[1,2], log10.(df.gradient), color=(:black, 0.2))
        lines!(ax[1,3], log10.(df.objective), color=(:black, 0.2))
    end
    for df in groupby(hist2, :replicate)
        lines!(ax[2,2], log10.(df.gradient), color=(:black, 0.2))
        lines!(ax[2,3], log10.(df.objective), color=(:black, 0.2))
    end

    colgap!(g, 5)
    rowgap!(g, 5)
    rowgap!(g, 3, 1)

    return fig
end


function cancer_size_distributions(dir, projection)
    examples = [
        "colon" "srbctA" "leukemiaA"
        "lymphomaA" "brain" "prostate"
    ]

    titles = [
        latexstring("Colon ", L"(p=2000)") latexstring("SRBCT ", L"(p=2308)") latexstring("Leukemia ", L"(p=3571)")
        latexstring("Lymphoma ", L"(p=4026)") latexstring("Brain ", L"(p=5597)") latexstring("Prostate ", L"(p=6033)")
    ]

    global SETTINGS
    fig = Figure(resolution=SETTINGS[:resolution], fontsize=SETTINGS[:fontsize])
    g = fig[1,1] = GridLayout()

    Label(g[3, 2:4], "Number of active features")
    Label(g[1:2, 1], "Frequency", rotation=pi/2)
    ax = [Axis(g[i,j+1], title=titles[i,j]) for i in 1:2, j in 1:3]
    linkyaxes!(ax...)

    for j in 1:3, i in 1:2
        _, modelA, _ = load_dfs(dir, examples[i,j], projection)
        hist!(ax[i,j], modelA.active_variables, bins=20)
    end

    colgap!(g, 5)
    rowgap!(g, 5)

    return fig
end

function TCGA_topgenes(dir, projection)
    global SETTINGS

    df = MVDA.dataset("TCGA-HiSeq")
    L, X = Vector{String}(df[!,1]), Matrix{Float64}(df[!,2:end])
    problem = MVDAProblem(L, X, intercept=false, encoding=:standard)
    class_reference = problem.labels
    n_classes = length(class_reference)

    gene_reference = CSV.read("/home/alanderos/Downloads/TCGA-PANCAN-HiSeq-genes.csv", DataFrame)[!,:symbol]
    selected_genes = CSV.read("/home/alanderos/.julia/datadeps/MVDA/TCGA-HiSeq.cols", DataFrame, header=false)[1:end-2,1]
    genes = gene_reference[selected_genes]
    common_kwargs = (;
        xticks=LinearTicks(6),
        yticklabelsize=SETTINGS[:fontsize],
    )

    find_nz(x) = findall(x -> norm(x) != 0, eachrow(x))
    coeff = [MVDA.load_model(joinpath(dir, "TCGA-HiSeq", projection, "modelA", "$(k)")).coeff_proj.slope for k in 1:10]

    resolution, factor = SETTINGS[:paper_size], SETTINGS[:resolution_scaling_factor]
    resolution = (n_classes * factor * 0.2 * 0.9 * resolution[1], factor * 0.6 * resolution[2])
    crange = (-5, 5)
    cmap = Reverse(:balance)

    fig = Figure(resolution=resolution, fontsize=SETTINGS[:fontsize])
    g = GridLayout(fig[1,1])
    ax = [Axis(g[1,j]; common_kwargs...) for j in 1:n_classes]
    for i in eachindex(ax)
        # Extract gene subsets of selected genes specific to class i; i.e. nonzero slopes
        effect_size = map(x -> vec(x[:,i]), coeff)
        nz_row = map(find_nz, effect_size)
        nz_row_idx = [sortperm(effect_size[k][nz_row[k]], by=abs, rev=true) for k in eachindex(nz_row)]
        gene_idx = [nz_row[k][nz_row_idx[k]] for k in eachindex(nz_row)]
        gene_subsets = map(idx -> genes[idx], gene_idx)

        # Count the number of times a gene is replicated
        cm = Dict{String,Int}()
        for subset in gene_subsets, gene in subset
            cm[gene] = 1 + get(cm, gene, 0)
        end
    
        # Repeat, but with array that will be used for weighted average
        nz_counts = zeros(Int, length(effect_size[1]))
        for arr in nz_row, j in arr
            nz_counts[j] += 1
        end
        replace!(nz_counts, 0 => 1)

        # Rank the genes from most replicated to least replicated
        gene_key, gene_count = collect(keys(cm)), collect(values(cm))
        idx = sortperm(gene_count, rev=true)
        gene_key, gene_count = gene_key[idx], gene_count[idx]
        topK = min(length(gene_count), 50)

        scaled_avg_effect_size = (1 ./ nz_counts) .* sum(effect_size)       # average of nonzero estimates only
        scaled_avg_effect_size = scaled_avg_effect_size[union(nz_row...)]   # nonzeros
        scaled_avg_effect_size = scaled_avg_effect_size[idx]                # sort by most replicated to least replicated
        scaled_avg_effect_size .*= 1e3                                      # scale up the effect sizes

        ylims!(ax[i], low=0, high=topK+1)
        barplot!(ax[i], gene_count[1:topK];
            color=scaled_avg_effect_size[1:topK],
            colormap=cmap,
            colorrange=crange,
            direction=:x,
        )
        ax[i].title = "$(class_reference[i])"       # TCGA cancer
        ax[i].yticks = (1:topK, gene_key[1:topK])   # label genes
        ax[i].yreversed = true                      # order from most replicated to least replicated
    end

    # Set common x-label
    Label(g[2, 1:end], "Frequency")

    # Set common colorbar
    Colorbar(g[3, 2:end-1];
        label=latexstring("Slope", L"\times 10^{3}"),
        colormap=cmap,
        colorrange=crange,
        vertical=false,
        flipaxis=false,
    )

    colgap!(g, 10)
    rowgap!(g, 10)

    return fig
end

##### Tables #####

function summarize_col(df, col, alpha, f)
    xs = df[!, col]
    lo = alpha/2
    hi = 1-lo
    v, l, h = map(f, (median(xs), quantile(xs, lo), quantile(xs, hi)))
    l, h = extrema((l, h))
    return (val=v, hi=h, lo=l)
end

error_percent_formatter(x) = round(100*(1-x), sigdigits=3)
number_formatter(x) = round(x, sigdigits=2)
number_formatter2(x) = round(x, sigdigits=3)
integer_formatter(x) = round(Int, x)

function make_comparative_table_by_row(dir, examples, projections; alpha=0.05)
    global ABBREVIATIONS
    io = IOBuffer()
    write(io, "\\begin{tabular}{ccccccc}\n")
    write(io, "\\toprule\n")
    write(io, "VDA Method & Train Error (\\%) & Test Error (\\%) & \$\\epsilon\$   & \$k\$ or \$\\lambda\$ & \\# Active Features & Time (s)   \\\\\n")
    write(io, "\\midrule\n")
    for example in examples
        write(io, "\\multicolumn{7}{c}{\\textbf{$(example)}} \\\\\n")
        write(io, "\\cmidrule(lr){2-7}\n")
        for projection in projections
            abbrv = ABBREVIATIONS[projection]
            if contains(abbrv, "L0")
                hyperparam = :k
                hyperparam_formatter = integer_formatter
            elseif contains(abbrv, "L1")
                hyperparam = :lambda
                hyperparam_formatter = number_formatter2
            end
            df_path, df_fit, _ = load_dfs(dir, example, projection)
            err_trn = summarize_col(df_fit, :train, alpha, error_percent_formatter)      # train error, %
            err_tst = summarize_col(df_fit, :test, alpha, error_percent_formatter)       # test error, %
            epsilon = summarize_col(df_fit, :epsilon, alpha, number_formatter2)          # deadzone
            hyparam = summarize_col(df_fit, hyperparam, alpha, hyperparam_formatter)     # k or lambda
            nactive = summarize_col(df_fit, :active_variables, alpha, integer_formatter) # number of active features

            tmp = combine(groupby(df_path, :replicate), :time => sum => :time)
            tmp.time .+= df_fit.time
            tcvtime = summarize_col(tmp, :time, alpha, number_formatter) # seconds; CV path + fit

            write(io, "$(abbrv)
                & $(err_trn.lo), $(err_trn.val), $(err_trn.hi)
                & $(err_tst.lo), $(err_tst.val), $(err_tst.hi)
                & $(epsilon.lo), $(epsilon.val), $(epsilon.hi)
                & $(hyparam.lo), $(hyparam.val), $(hyparam.hi)
                & $(nactive.lo), $(nactive.val), $(nactive.hi)
                & $(tcvtime.lo), $(tcvtime.val), $(tcvtime.hi) \\\\\n")
        end
    end
    write(io, "\\bottomrule\n")
    write(io, "\\end{tabular}")
    tbl = String(take!(io))
    close(io)
    return tbl
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
    summarize_col(data, col, alpha, fmt)
end

function make_comparative_table_by_col(dir, examples, projections, metrics; alpha=0.05)
    global ABBREVIATIONS
    
    abbrvs = map(x -> ABBREVIATIONS[x], projections)
    table_header = join(("    ", abbrvs...), " & ")
    table_header = string(table_header, " \\\\\n")
    ncols = length(projections)

    io = IOBuffer()
    col_header_str = repeat("c", ncols+1)
    write(io, "\\begin{tabular}{$(col_header_str)}\n")
    write(io, "\\toprule\n")
    write(io, table_header)
    write(io, "\\midrule\n")
    for example in examples
        write(io, "\\multicolumn{$(ncols+1)}{c}{\\textbf{$(example)}} \\\\\n")
        write(io, "\\cmidrule(lr){2-$(ncols+1)}\n")
        example_cols = Vector{String}[]

        # Row Labels
        metric_labels = String[]
        for metric in metrics
            push!(metric_labels, fetch_metric_label(metric))
        end
        push!(example_cols, metric_labels)

        for projection in projections
            abbrv = ABBREVIATIONS[projection]
            if contains(abbrv, "L0")
                hyperparam = :k
            elseif contains(abbrv, "L1")
                hyperparam = :lambda
            end
            df_path, df_fit, _ = load_dfs(dir, example, projection)
            coldata = String[]
            for metric in metrics
                if metric == :hyperparam
                    col = hyperparam
                else
                    col = metric
                end
                data = fetch_metric((df_path, df_fit), col, alpha)
                push!(coldata, "$(data.lo), $(data.val), $(data.hi)")
            end
            push!(example_cols, coldata)
        end
        for k in 1:length(metrics)
            rowdata = map(x -> x[k], example_cols)
            write(io, join(rowdata, " & "), "\\\\\n")
        end
    end
    write(io, "\\bottomrule\n")
    write(io, "\\end{tabular}")
    tbl = String(take!(io))
    close(io)
    return tbl
end

function make_comparative_table_by_example(dir, examples; alpha=0.05)
    global ABBREVIATIONS

    io = IOBuffer()

    write(io, "\\begin{tabular}{ccccccc}\n")
    write(io, "\\toprule\n")
    write(io, "    & \\multicolumn{2}{c}{HomL0} & \\multicolumn{2}{c}{HetL0} & \\multicolumn{2}{c}{HetL1} \\\\\n")
    write(io, "    \\cmidrule(lr){2-3}    \\cmidrule(lr){4-5}    \\cmidrule(lr){6-7}\n")
    write(io, "    & Error (\\%) & \\# Active & Error (\\%) & \\# Active & Error (\\%) & \\# Active \\\\\n")
    write(io, "\\midrule\n")

    for example in examples
        unused, df1, _ = load_dfs(dir, example, "HomogeneousL0Projection")
        data1_err = fetch_metric((unused, df1), :test, alpha)
        data1_act = fetch_metric((unused, df1), :active_variables, alpha)

        unused, df2, _ = load_dfs(dir, example, "HeterogeneousL0Projection")
        data2_err = fetch_metric((unused, df2), :test, alpha)
        data2_act = fetch_metric((unused, df2), :active_variables, alpha)

        unused, df3, _ = load_dfs(dir, example, "HeterogeneousL1BallProjection")
        data3_err = fetch_metric((unused, df3), :test, alpha)
        data3_act = fetch_metric((unused, df3), :active_variables, alpha)

        write(io, "$(example)
            & $(data1_err.lo), $(data1_err.val), $(data1_err.hi)
            & $(data1_act.lo), $(data1_act.val), $(data1_act.hi)
            & $(data2_err.lo), $(data2_err.val), $(data2_err.hi)
            & $(data2_act.lo), $(data2_act.val), $(data2_act.hi)
            & $(data3_err.lo), $(data3_err.val), $(data3_err.hi)
            & $(data3_act.lo), $(data3_act.val), $(data3_act.hi) \\\\\n")
    end

    write(io, "\\bottomrule\n")
    write(io, "\\end{tabular}")
    tbl = String(take!(io))
    close(io)
    return tbl
end

##### Execution of the script #####

function main(input, output)
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

    # Section 3.1
    @info "Generating Figure 2..."
    @time begin
        fig2 = synthetic_summary(uci_path)
        save(joinpath(figures, "Figure2.pdf"), fig2, pt_per_unit=1)
    end

    @info "Generating Figure 3..."
    @time begin
        fig3 = synthetic_coefficients(uci_path)
        save(joinpath(figures, "Figure3.pdf"), fig3, pt_per_unit=1)
    end

    @info "Generating Table 1..."
    @time begin
        tbl1 = make_comparative_table_by_col(uci_path,
            ("synthetic", "synthetic-hard"),
            (
                "HomogeneousL0Projection", "HeterogeneousL0Projection",
                "HomogeneousL1BallProjection", "HeterogeneousL1BallProjection",    
            ),
            [
                :train, :test, :epsilon, :hyperparam, :active_variables, :time,
            ]
        )
        save_table(joinpath(tables, "Table1.tex"), tbl1, nothing)
    end

    # Section 3.2
    @info "Generating Table 2..."
    @time begin
        tbl2 = make_comparative_table_by_example(cancer_path,
            (
                "colon", "srbctA", "leukemiaA", "lymphomaA", "brain",
                "prostate",
            );
            alpha=0.1
        )
        save_table(joinpath(tables, "Table2.tex"), tbl2, nothing)
    end
    
    @info "Generating Figure 4..."
    @time begin
        fig4 = cancer_size_distributions(cancer_path, "HeterogeneousL0Projection")
        save(joinpath(figures, "Figure4.pdf"), fig4, pt_per_unit=1)
    end

    # Section 3.3
    @info "Generating Table 3..."
    @time begin
        tbl3 = make_comparative_table_by_example(uci_path,
            (
                "iris", "lymphography", "zoo", "bcw","waveform",
                "splice", "letters", "optdigits", "vowel", "HAR",
                "TCGA-HiSeq",
            );
            alpha=0.1
        )
        save_table(joinpath(tables, "Table3.tex"), tbl3, nothing)
    end

    @info "Generating Figure 5..."
    @time begin
        fig5 = TCGA_topgenes(uci_path, "HeterogeneousL0Projection")
        save(joinpath(figures, "Figure5.pdf"), fig5, pt_per_unit=1)
    end

    # Section 3.4
    @info "Generating Table 4..."
    @time begin
        tbl4 = make_comparative_table_by_example(nonlinear_path,
            (
                "circles", "clouds", "waveform", "spiral", "spiral-hard",
                "vowel",
            );
            alpha=0.1
        )
        save_table(joinpath(tables, "Table4.tex"), tbl4, nothing)
    end

    return nothing
end

# Run the script.
input_dir, output_dir = ARGS[1], ARGS[2]
main(input_dir, output_dir)
