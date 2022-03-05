using CSV, DataFrames, LaTeXStrings, MVDA, Plots

# Global plot options
upscale = 1.0
font_main = Plots.font("Computer Modern", 18)
font_large = Plots.font("Computer Modern", 16)
font_small = Plots.font("Computer Modern", 12)
default()
default(
    annotationfontfamily="Computer Modern",
    markerstrokewidth=0.5,
    titlefont=font_main,
    guidefont=font_large,
    tickfont=font_small,
    legendfont=font_small,
    markersize=5,
    linewidth=3,
    dpi=200,
    thickness_scaling=1.0,
    top_margin=5Plots.mm,
)
scalefontsizes(upscale)

# Helper function to add a separate x-axis that emphasizes the number of features/support vectors in a model.
function add_model_size_guide(fig, N, is_linear)
    # Compute ticks based on maximum model size N.
    if N > 16
        xticks = collect(round.(Int, N .* range(0, 1, length=11)))
    else
        xticks = collect(0:N)
    end
    sort!(xticks, rev=true)

    xlabel = ifelse(is_linear, "no. features", "no. support vectors")

    # Register figure inside main subplot and append extra x-axis.
    model_size_guide = plot(yticks=nothing, xticks=xticks, xlim=(0,N), xlabel=xlabel, xflip=true)
    full_figure = plot(fig, model_size_guide, layout=@layout [a{1.0h}; b{1e-8h}])

    return full_figure
end

function make_plots(filename)
    # Common plot options.
    w, h = default(:size)
    options = (; left_margin=5Plots.mm, size=(1.5*w, 1.5*h),)

    # Read file containing CV results
    PROGRESS_ID = "make_plot"
    println()
    @info "Reading $(filename)" _id=PROGRESS_ID
    partial_filename = first(splitext(filename))
    cv_results = CSV.read(partial_filename*".dat", DataFrame, header=true)
    
    example = first(cv_results.example)
    data_subsets = first(cv_results.ntrain), first(cv_results.nvalidate), first(cv_results.ntest)
    p, c = first(cv_results.nfeatures), first(cv_results.nclasses)
    is_linear = first(cv_results.kernel) == "none"
    nvars = is_linear ? p : data_subsets[1]

    titles = ["$(example)\n$(_n) samples / $(p) features / $(c) classes" for _n in data_subsets]
    metrics = (:train, :validation, :test)

    # Summarize CV results over folds + make plot.
    @info "Summarizing over folds" _id=PROGRESS_ID
    cv_paths = MVDA.cv_error(cv_results)
    for (title, metric) in zip(titles, metrics)
        fig = MVDA.plot_cv_paths(cv_paths, metric)
        plot!(fig; title=title, options...)
        fig = add_model_size_guide(fig, nvars, is_linear)
        figname = partial_filename*"-replicates=$(metric).png"
        @info "Saving $(figname)" _id=PROGRESS_ID
        savefig(fig, figname)
    end

    # Construct credible intervals for detailed summary plot.
    @info "Constructing credible intervals" _id=PROGRESS_ID
    cv_intervals = MVDA.credible_intervals(cv_paths)
    for (title, metric) in zip(titles, metrics)
        fig = MVDA.plot_credible_intervals(cv_intervals, metric)
        plot!(fig; title=title, options...)
        fig = add_model_size_guide(fig, nvars, is_linear)
        figname = partial_filename*"-summary=$(metric).png"
        @info "Saving $(figname)" _id=PROGRESS_ID
        savefig(fig, figname)
    end
end

dir = ARGS[1]
@info "Reading from directory: $(dir)"
files = readdir(dir, join=true)
filter!(contains(".dat"), files)
for filename in files
    make_plots(filename)
end