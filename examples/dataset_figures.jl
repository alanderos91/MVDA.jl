using MVDA, Plots, Random, StableRNGs, LaTeXStrings

# Note: only show samples used in fitting and selecting models

w, h = default(:size)
default(
    left_margin=5Plots.mm,
    bottom_margin=5Plots.mm,
    size=(1.5*w, 1.5*h),
    dpi=200)
scalefontsizes(1.25)

make_figure = function(Y, X)
    scatter(X[:,1], X[:,2],
        group=Y,
        aspect_ratio=1,
        xlabel=L"feature $x_1$",
        ylabel=L"feature $x_2$",
        markersize=4,
        markerstrokewidth=1e-4,
    )
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

# Wang & Shen 2007
n_cv, n_test = 100, 10^3
nsamples = n_cv + n_test
nclasses = 4
p = 100

Y, X = MVDA.simulate_WS2007(nsamples, p, nclasses, round(Int, nsamples/nclasses), 1.0)
make_figure(Y[1:n_cv], X[1:n_cv,:])
title!(L"WS / $n=%$n_cv$ / $p=%$p$ / $c=%$nclasses$ / $d=1$")
savefig("/home/alanderos/Desktop/VDA/ws-under-hard.png")

Y, X = MVDA.simulate_WS2007(nsamples, p, nclasses, round(Int, nsamples/nclasses), 2.0)
make_figure(Y[1:n_cv], X[1:n_cv,:])
title!(L"WS / $n=%$n_cv$ / $p=%$p$ / $c=%$nclasses$ / $d=2$")
savefig("/home/alanderos/Desktop/VDA/ws-under-medium.png")

Y, X = MVDA.simulate_WS2007(nsamples, p, nclasses, round(Int, nsamples/nclasses), 3.0)
make_figure(Y[1:n_cv], X[1:n_cv,:])
title!(L"WS / $n=%$n_cv$ / $p=%$p$ / $c=%$nclasses$ / $d=3$")
savefig("/home/alanderos/Desktop/VDA/ws-under-easy.png")

n_cv, n_test = 500, 10^3
nsamples = n_cv + n_test
nclasses = 4
p = 100

Y, X = MVDA.simulate_WS2007(nsamples, p, nclasses, round(Int, nsamples/nclasses), 1.0)
make_figure(Y[1:n_cv], X[1:n_cv,:])
title!(L"WS / $n=%$n_cv$ / $p=%$p$ / $c=%$nclasses$ / $d=1$")
savefig("/home/alanderos/Desktop/VDA/ws-over-hard.png")

Y, X = MVDA.simulate_WS2007(nsamples, p, nclasses, round(Int, nsamples/nclasses), 2.0)
make_figure(Y[1:n_cv], X[1:n_cv,:])
title!(L"WS / $n=%$n_cv$ / $p=%$p$ / $c=%$nclasses$ / $d=2$")
savefig("/home/alanderos/Desktop/VDA/ws-over-medium.png")

Y, X = MVDA.simulate_WS2007(nsamples, p, nclasses, round(Int, nsamples/nclasses), 3.0)
make_figure(Y[1:n_cv], X[1:n_cv,:])
title!(L"WS / $n=%$n_cv$ / $p=%$p$ / $c=%$nclasses$ / $d=3$")
savefig("/home/alanderos/Desktop/VDA/ws-over-easy.png")

# Gaussian Clouds
n_cv, n_test = 250, 10^3
nsamples = n_cv + n_test
nclasses = 3
Y, X = MVDA.simulate_gaussian_clouds(nsamples, nclasses; sigma=0.25, rng=StableRNG(1903))
make_figure(Y[1:n_cv], X[1:n_cv,:])
draw_cloud_boundaries!()
title!(L"clouds / $n=%$n_cv$ / $p=2$ / $c=%$nclasses$")
savefig("/home/alanderos/Desktop/VDA/clouds.png")

# Nested Circles
n_cv, n_test = 250, 10^3
nsamples = n_cv + n_test
nclasses = 3
Y, X = MVDA.simulate_nested_circles(nsamples, nclasses; p=8//10, rng=StableRNG(1903))
make_figure(Y[1:n_cv], X[1:n_cv,:])
title!(L"clouds / $n=%$n_cv$ / $p=2$ / $c=%$nclasses$")
foreach(r -> draw_circle!(sqrt(r)), 1:nclasses)
savefig("/home/alanderos/Desktop/VDA/circles.png")

# Waveform
n_cv, n_test = 375, 10^3
nsamples = n_cv + n_test
nfeatures = 21
Y, X = MVDA.simulate_waveform(nsamples, nfeatures; rng=StableRNG(1903))
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
savefig("/home/alanderos/Desktop/VDA/waveform.png")
