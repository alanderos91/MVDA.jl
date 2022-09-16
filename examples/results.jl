using MVDA, CairoMakie, CSV, DataFrames, Statistics, StatsBase, Latexify, LaTeXStrings, LinearAlgebra

SETTINGS = Dict(:resolution => 72 .* (8.5*0.9, 11*0.3), :fontsize => 8)

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
        limits=(-1, nvars+1, 0, 1),
        yticks = LinearTicks(5),
        kwargs...
    )

    plot_cv_path!(ax, path_df, nvars)

    resize_to_layout!(fig)
    return fig
end

function plot_cv_path!(ax, path_df, nvars)
    path_gdf = groupby(path_df, [:title, :algorithm, :replicate, :sparsity])
    score_df = combine(path_gdf, :validation => mean => identity)

    for df in groupby(score_df, :replicate)
        xs = round.(Int, nvars * (1 .- df.sparsity))
        ys = 1 .- df.validation
        lines!(ax, xs, ys, color=(:black, 0.1), linewidth=3)
    end

    avg_path = combine(groupby(score_df, :sparsity), :validation => mean => identity)
    xs = round.(Int, nvars * (1 .- avg_path.sparsity))
    ys = 1 .- avg_path.validation
    lines!(ax, xs, ys, color=:red, linewidth=2, linestyle=:dash)

    return ax
end

load_dfs(dir, example) = (
    CSV.read(joinpath(dir, example, "cv_path.out"), DataFrame),
    CSV.read(joinpath(dir, example, "modelA", "summary.out"), DataFrame),
    CSV.read(joinpath(dir, example, "modelB", "summary.out"), DataFrame)
)

function load_replicate_data(dir, example, model, filename)
    example_dir = joinpath(dir, example, model)
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

load_hist(dir, example, model) = load_replicate_data(dir, example, model, "history.out")
load_cmat(dir, example, model) = load_replicate_data(dir, example, model, "confusion_matrix.out")
load_prob(dir, example, model) = load_replicate_data(dir, example, model, "probability_matrix.out")

function load_coefficients(dir, example, model)
    example_dir = joinpath(dir, example, model)
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

function save_table(filepath, df, header)
    table_str = latexify(df,
        latex=false,
        env=:table,
        booktabs=true,
        head=header,
        adjustment=:l
    )
    write(filepath, table_str)
end

##### Figures #####

function synthetic_coefficients(dir)
    mse(x, y) = mean( (x .- y) .^ 2 )

    arrA = load_coefficients(dir, "synthetic", "modelA")
    arrB = load_coefficients(dir, "synthetic", "modelB")

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
    path_df1, _, _ = load_dfs(dir, "synthetic")
    path_df2, _, _ = load_dfs(dir, "synthetic-hard")
    
    hist1 = load_hist(dir, "synthetic", "modelA")
    hist2 = load_hist(dir, "synthetic-hard", "modelA")

    nvars = 500
    kwargs = (;
        xreversed=true,
        xticks = LinearTicks(min(11, nvars)),
        limits=(-1, nvars+1, 0, 1),
        yticks = LinearTicks(5),
    )

    global SETTINGS
    fig = Figure(resolution=SETTINGS[:resolution], fontsize=SETTINGS[:fontsize])
    g = fig[1,1] = GridLayout()

    Box(g[1,2:3], color=:gray90, tellheight=false)
    Label(g[1,2:3], "Classification error", tellwidth=false)
    
    Box(g[1,4], color=:gray90, tellheight=false)
    Label(g[1,4], "Gradient norm", tellwidth=false)

    Box(g[1,5], color=:gray90, tellheight=false)
    Label(g[1,5], "Objective", tellwidth=false)

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


function cancer_size_distributions(dir)
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
        _, modelA, _ = load_dfs(dir, examples[i,j])
        hist!(ax[i,j], modelA.active_variables, bins=20)
    end

    colgap!(g, 5)
    rowgap!(g, 5)

    return fig
end

function TCGA_topgenes()
    gene_reference = CSV.read("/home/alanderos/Downloads/TCGA-PANCAN-HiSeq-genes.csv", DataFrame)[!,:symbol]
    selected_genes = CSV.read("/home/alanderos/.julia/datadeps/MVDA/TCGA-HiSeq.cols", DataFrame, header=false)[1:end-2,1]
    genes = gene_reference[selected_genes]

    find_nz(x) = findall(x -> norm(x) != 0, eachrow(x))
    coeff = [MVDA.load_model("/home/alanderos/Desktop/VDA-Results/linear/TCGA-HiSeq/modelA/$(k)").coeff_proj.slope for k in 1:10]
    effect_size = map(x -> map(norm, eachrow(x)), coeff)
    nz_row = map(find_nz, coeff)
    gene_idx = [sortperm(effect_size[k][nz_row[k]], rev=true) for k in eachindex(nz_row)]

    gene_subsets = map(idx -> genes[idx], gene_idx)

    cm = Dict{String,Int}()
    for subset in gene_subsets, gene in subset
        cm[gene] = 1 + get(cm, gene, 0)
    end

    gene_key, gene_count = collect(keys(cm)), collect(values(cm))
    idx = sortperm(gene_count, rev=true)
    gene_key, gene_count = gene_key[idx], gene_count[idx]

    topK = min(length(gene_count), 50)

    global SETTINGS
    fig = Figure(resolution=SETTINGS[:resolution], fontsize=SETTINGS[:fontsize])
    ax = Axis(fig[1,1], xticks=(1:topK, gene_key[1:topK]), xticklabelrotation=pi/6, ylabel="Frequency", yticks=LinearTicks(6))
    barplot!(ax, gene_count[1:topK],
        # bar_labels=gene_count[1:topK],
        # label_size=SETTINGS[:fontsize],
    )

    return fig
end

function MS_error_rates(dir)
    #
    function load_and_convert(dir, example, idx)
        df = load_cmat(dir, example, "modelA")
        filter!(:subset => isequal("test"), df)
        gdf = groupby(df, :replicate)
        C = [Matrix{Float64}(gdf[k][!,idx]) for k in eachindex(gdf)]
        P = map(Ck -> Ck ./ sum(Ck, dims=2), C)
        return P
    end
    #
    function plot_confusion_matrix!(ax, data)
        heatmap!(ax, data, colormap=:Blues, colorrange=(0,1))
        str = vec(map(x -> string(round(x, sigdigits=3)), data))
        pos = vec([(i,j) for i in axes(data, 1), j in axes(data, 2)])
        text!(ax, str,
            position=pos,
            textsize=8,
            align=(:center,:center),
            color=map(x -> x/sum(data) > 1/length(data) ? :white : :black, vec(data)),
        )
    end
    #
    examples = ["all", "dropmissing", "combined", "features"]
    labels = ["Full dataset" "Drop missing"; "Combine B and C" "Filter features"]
    classes = ["A", "B", "C"]
    kwargs = (;
        xlabel="Class",
        ylabel="Predicted",
        xticks=(1:3, classes),
        yticks=(1:3, classes),
        xticksize=0,
        yticksize=0,
        yreversed=true,
    )

    xs, ys, dodge = Int[], Float64[], Int[]

    for (k, example) in enumerate(examples)
        _, modelA, modelB = load_dfs(dir, example)
        for (accA, accB) in zip(modelA.test, modelB.test)
            # sparse model
            push!(xs, k)
            push!(ys, 1-accA)
            push!(dodge, 1)
            # reduced model
            push!(xs, k)
            push!(ys, 1-accB)
            push!(dodge, 2)
        end
    end

    modelA = load_and_convert(dir, "all", 3:5)
    modelB = load_and_convert(dir, "dropmissing", 3:5)
    modelC = load_and_convert(dir, "combined", 3:4)
    modelD = load_and_convert(dir, "features", 3:5)

    global SETTINGS
    fig = Figure(resolution=SETTINGS[:resolution], fontsize=SETTINGS[:fontsize])
    g = fig[1,1] = GridLayout()

    axmain = Axis(g[1:2,1:2],
        xticks=(1:4, ["Full\ndataset", "Drop\nmissing", "Combine\nB and C", "Filter\nfeatures"]),
        ylabel="Classification error",
        xticksize=0,
        yticksize=2,
        xticklabelalign=(:center,:top),
    )
    ax = [Axis(g[i,j+2]; title=labels[i,j], kwargs...) for i in 1:2, j in 1:2]

    boxplot!(axmain, xs, ys,
        dodge=dodge,
        color=map(d -> d==1 ? :gray75 : :gray50, dodge),
        shownotch=true,
        width=0.6,
    )
    Legend(
        g[1:2,1:2],
        [PolyElement(color=:gray75), PolyElement(color=:gray50)],
        ["Sparse", "Reduced"],
        framevisible=false,
        halign=:center,
        valign=:top,
        rowgap=5,
    )
    
    plot_confusion_matrix!(ax[1,1], mean(modelA))
    plot_confusion_matrix!(ax[1,2], mean(modelB))
    plot_confusion_matrix!(ax[2,1], mean(modelC))
    plot_confusion_matrix!(ax[2,2], mean(modelD))

    colgap!(g, 5)
    rowgap!(g, 5)

    return fig
end

##### Tables #####

function benchmark_summary(dir, examples; include_gamma=false)
    #
    function summarize_col(df, col, alpha, f)
        xs = df[!, col]
        lo = alpha/2
        hi = 1-lo
        return map(f, (median(xs), quantile(xs, lo), quantile(xs, hi)))
    end
    #
    function write_ci(x, lo, hi)
        latexstring("$(x)\\, ($(lo), $(hi))")
    end
    #
    number_formatter(x) = round(x, sigdigits=2)
    number_formatter2(x) = round(x, sigdigits=3)
    integer_formatter(x) = round(Int, x)

    title = String[]
    es = LaTeXString[]
    ls = LaTeXString[]
    gs = LaTeXString[]
    ks = LaTeXString[]
    trn = LaTeXString[]
    tst = LaTeXString[]

    alpha = 0.05

    for example in examples
        _, summary_df, _ = load_dfs(dir, example)

        # use log10 scale for epsilon and lambda
        summary_df.epsilon .= log10.(summary_df.epsilon)
        summary_df.lambda .= log10.(summary_df.lambda)

        # use error instead of accuracy
        summary_df.train .= 1 .- summary_df.train
        summary_df.test .= 1 .- summary_df.test

        e, e_lo, e_hi = summarize_col(summary_df, :epsilon, alpha, number_formatter2)
        l, l_lo, l_hi = summarize_col(summary_df, :lambda, alpha, number_formatter2)
        k, k_lo, k_hi = summarize_col(summary_df, :active_variables, alpha, integer_formatter)
        train, train_lo, train_hi = summarize_col(summary_df, :train, alpha, number_formatter)
        test, test_lo, test_hi = summarize_col(summary_df, :test, alpha, number_formatter)

        push!(title, first(summary_df.title))
        push!(es, write_ci(e, e_lo, e_hi))
        push!(ls, write_ci(l, l_lo, l_hi))
        push!(ks, write_ci(k, k_lo, k_hi))
        push!(trn, write_ci(train, train_lo, train_hi))
        push!(tst, write_ci(test, test_lo, test_hi))

        if include_gamma
            summary_df.gamma .= log10.(summary_df.gamma)
            g, g_lo, g_hi = summarize_col(summary_df, :gamma, alpha, number_formatter2)
            push!(gs, write_ci(g, g_lo, g_hi))
        end
    end

    if include_gamma
        df = DataFrame(title=title, epsilon=es, lambda=ls, gamma=gs, k=ks, train=trn, test=tst)
    else
        df = DataFrame(title=title, epsilon=es, lambda=ls, k=ks, train=trn, test=tst)
    end

    return df
end

function uci_benchmarks(dir)
    examples = ["iris", "lymphography", "zoo", "bcw", "waveform", "splice", "letters", "optdigits", "vowel", "HAR", "TCGA-HiSeq"]
    df = benchmark_summary(dir, examples)
    return df
end

function cancer_benchmarks(dir)
    examples = ["colon", "srbctA", "leukemiaA", "lymphomaA", "brain", "prostate"]
    df = benchmark_summary(dir, examples)
    return df
end

function nonlinear_benchmarks(dir)
    examples = ["circles", "clouds", "waveform", "spiral", "spiral-hard", "vowel"]
    df = benchmark_summary(dir, examples, include_gamma=false)
    return df
end

##### Execution of the script #####

function main(input, output)
    uci_path = joinpath(input, "linear")
    cancer_path = joinpath(input, "cancer")
    nonlinear_path = joinpath(input, "nonlinear")
    ms_path = joinpath(input, "MS")

    figures = joinpath(output, "figures")
    tables = joinpath(output, "tables")
    for dir in (figures, tables)
        if !ispath(dir)
            mkpath(dir)
        end
    end

    header = [
        "",
        L"\multicolumn{1}{c}{$\log_{10}(\epsilon)$}",
        L"\multicolumn{1}{c}{$\log_{10}(\lambda)$}",
        L"\multicolumn{1}{c}{$k$}",
        "\\multicolumn{1}{c}{Train}",
        "\\multicolumn{1}{c}{Test}",
    ]
    # header_gamma = [
    #     header[1:3];
    #     [L"\multicolumn{1}{c}{$\log_{10}(\gamma)$}"];
    #     header[4:end]    
    # ]

    # Sec 3.1
    fig2 = synthetic_summary(uci_path)
    save(joinpath(figures, "Figure2.pdf"), fig2, pt_per_unit=1)

    fig3 = synthetic_coefficients(uci_path)
    save(joinpath(figures, "Figure3.pdf"), fig3, pt_per_unit=1)

    fig4 = TCGA_topgenes()
    save(joinpath(figures, "Figure4.pdf"), fig4, pt_per_unit=1)

    # Sec 3.2
    df1 = uci_benchmarks(uci_path)
    save_table(joinpath(tables, "Table1.tex"), df1, header)

    # Sec 3.3
    df2 = nonlinear_benchmarks(nonlinear_path)
    save_table(joinpath(tables, "Table2.tex"), df2, header)

    # Sec 3.4
    df3 = cancer_benchmarks(cancer_path)
    save_table(joinpath(tables, "Table3.tex"), df3, header)
    
    fig5 = cancer_size_distributions(cancer_path)
    save(joinpath(figures, "Figure5.pdf"), fig5, pt_per_unit=1)

    # Sec 3.5
    fig6 = MS_error_rates(ms_path)
    save(joinpath(figures, "Figure6.pdf"), fig6, pt_per_unit=1)

    return
end

# Run the script.
input_dir, output_dir = ARGS[1], ARGS[2]
main(input_dir, output_dir)
