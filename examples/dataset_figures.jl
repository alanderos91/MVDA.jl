using MVDA, Plots, Random, StableRNGs, LaTeXStrings

# Note: only show samples used in fitting and selecting models

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
    size=(1.5*w, 1.5*h),
    markersize=5,
    linewidth=3,
    dpi=200,
    thickness_scaling=1.0,
    left_margin=5Plots.mm,
    bottom_margin=5Plots.mm,
)
scalefontsizes(upscale)

make_figure = function(Y, X, is2D::Bool=true)
    if is2D
        scatter(X[:,1], X[:,2],
            group=Y,
            aspect_ratio=1,
            xlabel=L"feature $x_1$",
            ylabel=L"feature $x_2$",
            markersize=4,
            markerstrokewidth=1e-4,
        )
    else
        scatter(X[:,1], X[:,2], X[:,3],
            group=Y,
            aspect_ratio=1,
            xlabel=L"feature $x_1$",
            ylabel=L"feature $x_2$",
            zlabel=L"feature $x_3$",
            markersize=4,
            markerstrokewidth=1e-4,
        )
    end
end

draw_cloud_boundaries! = function() # assumes 3 classes
    xs = range(-2,2,length=11)
    plot!(xs, x -> sqrt(2)/2*x, color=:black, style=:dash, label=nothing)
    plot!(xs, x -> -sqrt(2)/2*x, color=:black, style=:dash, label=nothing)
    vline!([0.0], color=:black, style=:dash, label=nothing)
end

draw_circle! = function(r)
    xs = range(-r, r, length=100)
    ys = sqrt.(r^2 .- xs .^2)
    plot!(xs, ys, color=:black, style=:dash, label="", lw=2)
    plot!(xs, -ys, color=:black, style=:dash, label="", lw=2)
end

length(ARGS) < 1 && error("Script requires an argument to specify a target directory.")
dir = ARGS[1]

# VDA Simulation
n_cv, n_test = 500, 500
nsamples = n_cv + n_test
p = 50
c = 10
samples_per_class = nsamples ÷ c
rng = StableRNG(1234)

Y, X, _ = MVDA.simulate_ground_truth(p, c, samples_per_class, d=1.0, rng=rng, sigma=1.0)
fig_filename = joinpath(dir, "10clouds-1.png")
make_figure(Y[1:n_cv], X[1:n_cv,:], false)
title!(L"10clouds / $n=%$n_cv$ / $p=%$p$ / $c=%$c$ / $d=1$")
@info "Saving $(fig_filename)"
savefig(fig_filename)

Y, X, _ = MVDA.simulate_ground_truth(p, c, samples_per_class, d=1.0, rng=rng, sigma=5.0)
fig_filename = joinpath(dir, "10clouds-2.png")
make_figure(Y[1:n_cv], X[1:n_cv,:], false)
title!(L"10clouds / $n=%$n_cv$ / $p=%$p$ / $c=%$c$ / $d=1$")
@info "Saving $(fig_filename)"
savefig(fig_filename)

Y, X, _ = MVDA.simulate_ground_truth(p, c, samples_per_class, d=1.0, rng=rng, sigma=10.0)
fig_filename = joinpath(dir, "10clouds-3.png")
make_figure(Y[1:n_cv], X[1:n_cv,:], false)
title!(L"10clouds / $n=%$n_cv$ / $p=%$p$ / $c=%$c$ / $d=1$")
@info "Saving $(fig_filename)"
savefig(fig_filename)

Y, X, _ = MVDA.simulate_ground_truth(p, c, samples_per_class, d=3.0, rng=rng, sigma=1.0)
fig_filename = joinpath(dir, "10clouds-4.png")
make_figure(Y[1:n_cv], X[1:n_cv,:], false)
title!(L"10clouds / $n=%$n_cv$ / $p=%$p$ / $c=%$c$ / $d=3$")
@info "Saving $(fig_filename)"
savefig(fig_filename)

Y, X, _ = MVDA.simulate_ground_truth(p, c, samples_per_class, d=3.0, rng=rng, sigma=5.0)
fig_filename = joinpath(dir, "10clouds-5.png")
make_figure(Y[1:n_cv], X[1:n_cv,:], false)
title!(L"10clouds / $n=%$n_cv$ / $p=%$p$ / $c=%$c$ / $d=3$")
@info "Saving $(fig_filename)"
savefig(fig_filename)

Y, X, _ = MVDA.simulate_ground_truth(p, c, samples_per_class, d=3.0, rng=rng, sigma=10.0)
fig_filename = joinpath(dir, "10clouds-6.png")
make_figure(Y[1:n_cv], X[1:n_cv,:], false)
title!(L"10clouds / $n=%$n_cv$ / $p=%$p$ / $c=%$c$ / $d=3$")
@info "Saving $(fig_filename)"
savefig(fig_filename)

# Gaussian Clouds
n_cv, n_test = 250, 10^3
nsamples = n_cv + n_test
nclasses = 3
Y, X = MVDA.simulate_gaussian_clouds(nsamples, nclasses; sigma=0.25, rng=StableRNG(1903))
fig_filename = joinpath(dir, "clouds.png")
make_figure(Y[1:n_cv], X[1:n_cv,:])
draw_cloud_boundaries!()
title!(L"clouds / $n=%$n_cv$ / $p=2$ / $c=%$nclasses$")
@info "Saving $(fig_filename)"
savefig(fig_filename)

# Nested Circles
n_cv, n_test = 250, 10^3
nsamples = n_cv + n_test
nclasses = 3
Y, X = MVDA.simulate_nested_circles(nsamples, nclasses; p=8//10, rng=StableRNG(1903))
fig_filename = joinpath(dir, "circles.png")
make_figure(Y[1:n_cv], X[1:n_cv,:])
title!(L"clouds / $n=%$n_cv$ / $p=2$ / $c=%$nclasses$")
foreach(r -> draw_circle!(sqrt(r)), 1:nclasses)
@info "Saving $(fig_filename)"
savefig(fig_filename)

# Waveform
n_cv, n_test = 375, 10^3
nsamples = n_cv + n_test
nfeatures = 21
Y, X = MVDA.simulate_waveform(nsamples, nfeatures; rng=StableRNG(1903))
fig_filename = joinpath(dir, "waveform.png")
Y, X = Y[1:n_cv], X[1:n_cv,:]
fig = plot(layout=@layout[a{1e-6h}; grid(3,1)])
title!(L"waveform / $n=%$n_cv$ / $p=21$ / $c=3$", subplot=1)
plot!(framestyle=:none, xticks=nothing, yticks=nothing, subplot=1)
@views for c in 1:3
    idx = findall(isequal(c), Y)
    foreach(i -> plot!(axes(X, 2), X[i,:], color=c, label=nothing, title="Class $c", alpha=0.25, subplot=c+1), idx)
end
ylabel!(L"feature $x_{j}$", subplot=3)
xlabel!(L"index $j$", subplot=4)
xticks!(1:nfeatures)
@info "Saving $(fig_filename)"
savefig(fig_filename)
