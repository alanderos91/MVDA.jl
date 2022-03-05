using CSV, DataFrames, LaTeXStrings, Plots

function add_model_size_guide(fig, N, flag)
    # Compute ticks based on maximum model size N.
    if N > 16
        xticks = collect(round.(Int, N .* range(0, 1, length=11)))
    else
        xticks = collect(0:N)
    end
    sort!(xticks, rev=flag)

    # Register figure inside main subplot and append extra x-axis.
    model_size_guide = plot(
        yticks=nothing,
        xticks=xticks,
        xlims=(-1,N+1),
        xlabel=latexstring("\\mathrm{target~number~of~features,}", "k"),
        xflip=flag,
        margins=10Plots.mm,
    )
    full_figure = plot(fig, model_size_guide, layout=@layout [a{1.0h}; b{1e-8h}])

    return full_figure
end

function __plot_accuracy__(fig, xs, ys, n, d, σ, marker, color)
    m = marker[d]
    ms = default(:markersize)
    plot!(fig, xs, ys,
        title=latexstring("n=$(n)"),
        xlabel="target sparsity (%)",
        ylabel="accuracy (%)",
        xticks=0:10:100,
        yticks=0:10:100,
        xlims=(-5,105),
        ylims=(-5,105),
        label=latexstring("d=$(d), \\sigma=$(σ)"),
        linestyle=:dash,
        marker=m,
        markersize=ifelse(m == :star, 1.25*ms, ms),
        color=color[σ],
        legend=false,
    )
end

function __plot_mse__(fig, xs, ys, n, d, σ, marker, color)
    m = marker[d]
    ms = default(:markersize)
    lower_log_lim = -6
    lower_lim = 10.0 ^ lower_log_lim
    plot!(fig, xs, ys,
        title=latexstring("n=$(n)"),
        xlabel="target sparsity (%)",
        ylabel=latexstring("\\mathrm{MSE}~(\\mathbf{B}_{k}, \\mathbf{B}_{0})"),
        xticks=0:10:100,
        yticks=10.0 .^ range(lower_log_lim, 0, step=1),
        yscale=:log10,
        xlims=(-5,105),
        ylims=(lower_lim,1e0),
        label=latexstring("d=$(d), \\sigma=$(σ)"),
        linestyle=:dash,
        marker=m,
        markersize=ifelse(m == :star, 1.25*ms, ms),
        color=color[σ],
        legend=false,
    )
end

function __plot_false_rate__(fig, xs, ys, n, d, σ, is_positive, marker, color)
    m = marker[d]
    ms = default(:markersize)
    plot!(fig, xs, ys,
        title=latexstring("n=$(n)"),
        xlabel="target sparsity (%)",
        ylabel=ifelse(is_positive, "false positive rate (%)", "false negative rate (%)"),
        xticks=0:10:100,
        yticks=0:10:100,
        xlims=(-5,105),
        ylims=(-5,105),
        label=latexstring("d=$(d), \\sigma=$(σ)"),
        linestyle=:dash,
        marker=m,
        markersize=ifelse(m == :star, 1.25*ms, ms),
        color=color[σ],
        # legend=ifelse(is_positive, :topright, :topleft),
        legend=false,
    )
end

function __plot_roc__(fig, xs, ys, n, d, σ, marker, color)
    m = marker[d]
    ms = default(:markersize)
    common_ticks = (0:10:100, 0:10:100)
    plot!(fig, xs, ys,
        title=latexstring("n=$(n)"),
        xlabel="false positive rate (%)",
        ylabel="true positive rate (%)",
        xticks=common_ticks,
        yticks=common_ticks,
        label=latexstring("d=$(d), \\sigma=$(σ)"),
        linestyle=:dash,
        marker=m,
        markersize=ifelse(m == :star, 1.25*ms, ms),
        color=color[σ],
        legend=false,
    )
end

function run(filename, outputdir)
    w, h = default(:size)
    upscale = 1.0
    font_main = Plots.font("Computer Modern", 24)
    font_large = Plots.font("Computer Modern", 16)
    font_small = Plots.font("Computer Modern", 12)
    default()
    default(
        markerstrokewidth=0.5,
        titlefont=font_main,
        guidefont=font_large,
        tickfont=font_small,
        legendfont=font_small,
        size=(upscale*w, upscale*h),
        markersize=5,
        linewidth=3,
        dpi=200,
        thickness_scaling=1.0,
    )
    scalefontsizes(upscale)

    df = CSV.read(filename, DataFrame, header=true)
    
    sort!(df, [:separation, :sigma])

    dlevels = levels(df.separation)
    σlevels = levels(df.sigma)
    slevels = levels(df.sparsity)

    primary_columns = [:n, :p, :c, :nclass, :ncausal]
    secondary_columns = [:separation, :sigma]

    markers = repeat([:star, :utriangle, :circle], length(dlevels))
    ms = Dict(d => m for (d, m) in zip(dlevels, markers))

    colors = palette(:seaborn_colorblind)
    cs = Dict(σ => c for (σ, c) in zip(σlevels, colors))

    M = 1
    idx = 1:M:length(slevels)

    for gdf in groupby(df, primary_columns)
        p = gdf.p[1]
        n = gdf.n[1]
        ncausal = gdf.ncausal[1]

        # Initialize plots
        figA, figB, figC, figD, figE = plot(), plot(), plot(), plot(), plot()
        append_accuracy!(xs, ys, d, σ) = __plot_accuracy__(figA, xs, ys, n, d, σ, ms, cs)
        append_mse!(xs, ys, d, σ) = __plot_mse__(figB, xs, ys, n, d, σ, ms, cs)
        append_false_positives!(xs, ys, d, σ) = __plot_false_rate__(figC, xs, ys, n, d, σ, true, ms, cs)
        append_false_negatives!(xs, ys, d, σ) = __plot_false_rate__(figD, xs, ys, n, d, σ, false, ms, cs)
        append_roc!(xs, ys, d, σ) = __plot_roc__(figE, xs, ys, n, d, σ, ms, cs)

        for sdf in groupby(gdf, secondary_columns)
            # Retrieve values used for grouping.
            d, σ = sdf.separation[1], sdf.sigma[1]

            # Select a subset of the data to avoid plot density.
            tmp = sdf[idx, :]
            sparsity = 100 * tmp.sparsity
            classification_accuracy = 100 .- tmp.error
            mse = tmp.MSE .+ eps()
            false_positive = 100 * tmp.FP ./ (tmp.FP .+ tmp.TN)
            false_negative = 100 * tmp.FN ./ (tmp.FN .+ tmp.TP)

            # Avoid plotting false positive/negatives when sparsity = 0%.
            # This part of the solution path is not well-defined.
            # Pad array with repeated value to make markers show up in GR backend...
            false_negative[1] = false_positive[1] = NaN
            insert!(false_positive, 1, false_positive[2])
            insert!(false_negative, 1, false_negative[2])
            padded_sparsity = [sparsity[2]; sparsity]

            # Fig A: Error vs Sparsity
            append_accuracy!(sparsity, classification_accuracy, d, σ)

            # Fig B: MSE vs Sparsity
            append_mse!(sparsity, mse, d, σ)

            # Fig C: FP vs Sparsity
            append_false_positives!(padded_sparsity, false_positive, d, σ)

            # Fig D: FN vs Sparsity
            append_false_negatives!(padded_sparsity, false_negative, d, σ)

            # Fig E: ROC
            append_roc!(false_positive, 100 .- false_negative, d, σ)
        end

        legend_panel = plot(deepcopy(figA),
            framestyle=:none,
            title="",
            legend=true,
            xlims=(-1,-0.5),
            ylims=(-1,-0.5),
            aspect_ratio=1000,
        )

        # Add size guides
        figA = add_model_size_guide(figA, p, true)
        figB = add_model_size_guide(figB, p, true)
        figC = add_model_size_guide(figC, p, true)
        figD = add_model_size_guide(figD, p, true)
        figE = add_model_size_guide(figE, p, false)
        
        # Add dashed y=x line on ROC curve
        plot!(figE, 0:10:100, x->x, label="", color=:black, ls=:dash)
        foreach(fig -> vline!(fig, [100*(1-ncausal/p)], color=:black, ls=:dot, label=""), (figA, figB, figC, figD))
        
        fig_filename = joinpath(outputdir, "n=$(n)-accuracy.png")
        @info "Saving $(fig_filename)"
        savefig(figA, fig_filename)

        fig_filename = joinpath(outputdir, "n=$(n)-mse.png")
        @info "Saving $(fig_filename)"
        savefig(figB, fig_filename)

        fig_filename = joinpath(outputdir, "n=$(n)-fp.png")
        @info "Saving $(fig_filename)"
        savefig(figC, fig_filename)

        fig_filename = joinpath(outputdir, "n=$(n)-fn.png")
        @info "Saving $(fig_filename)"
        savefig(figD, fig_filename)

        fig_filename = joinpath(outputdir, "n=$(n)-roc.png")
        @info "Saving $(fig_filename)"
        savefig(figE, fig_filename)

        fig_filename = joinpath(outputdir, "n=$(n)-legend.png")
        @info "Saving $(fig_filename)"
        savefig(legend_panel, fig_filename)
    end

    return nothing
end

# Parse input arguments and run.
length(ARGS) < 2 && error("Script requires (1) an input filename and (2) an output directory.")
filename = ARGS[1]
outputdir = ARGS[2]
@info "Input filename: $(filename)"
@info "Output directory: $(outputdir)"
run(filename, outputdir)
