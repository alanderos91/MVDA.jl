using CSV, DataFrames, Plots, StatsPlots, MVDA, LaTeXStrings

default()
w, h = default(:size)
upscale = 1.0
font_main = Plots.font("Computer Modern", 24)
font_large = Plots.font("Computer Modern", 16)
font_small = Plots.font("Computer Modern", 12)
default(
    markerstrokewidth=0.5,
    titlefont=font_main,
    guidefont=font_large,
    tickfont=font_small,
    legendfont=font_small,
    size=(upscale*w, upscale*h),
    dpi=200,
    thickness_scaling=1.0,
    palette=palette(:seaborn_colorblind),
)
scalefontsizes(upscale)

function optimal_model(df, p)
    f = map(r -> (r.validation, p*(100-r.sparsity)*1e-2), eachrow(df))
    j = argmin(f)
    return clamp(round(Int, last(f[j])), 0, p)
end

function model_density_plot(filename)
    results = CSV.read(filename, DataFrame)
    example = first(results.example)
    p = first(results.nfeatures)
    gdf = groupby(MVDA.cv_error(results), :replicate)
    xs = map(i -> optimal_model(gdf[i], p), eachindex(gdf))
    histogram(xs, normalize=true, lw=1e-2, label=nothing, alpha=0.8)
    density!(xs, xlabel="no. active features", title=L"%$example ($p = %$p$)", label=nothing, trim=false, normalize=true, lw=3, margins=5Plots.mm,)
end

dir = ARGS[1]
@info "Reading from directory: $(dir)"
files = readdir(dir, join=true)
filter!(contains(".dat"), files)
for file in files
    partial_filename = first(splitext(file))
    figure_file = partial_filename*"-feature-selection.png"
    @info "Reading results from $file"
    fig = model_density_plot(file)
    @info "Saving $(figure_file)"
    savefig(figure_file)
end