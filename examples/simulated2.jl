using MVDA, Plots, Statistics, Random, LinearAlgebra

function draw_circle!(v, ϵ)
    xs = v[1] .+ range(-ϵ, ϵ, length=201)
    ys1 = sqrt.(ϵ^2 .- (xs .- v[1]) .^2) .+ v[2]
    ys2 = -sqrt.(ϵ^2 .- (xs .- v[1]) .^2) .+ v[2]
    plot!(xs, ys1, color=:black, style=:dash, label="")
    plot!(xs, ys2, color=:black, style=:dash, label="")
end

# problem dimensions
nsamples = 100
c = 4
n = nsamples*c
p = 100

# simulate 4 classes with single predictor as in Wang and Shen 2007
overlap = 1.0
X = randn(n, p)
for i in 1:c
    a₁ = overlap * cos(2*(i-1)*π/c)
    a₂ = overlap * sin(2*(i-1)*π/c)
    X[nsamples*(i-1)+1:nsamples*i,1] .+= a₁
    X[nsamples*(i-1)+1:nsamples*i,2] .+= a₂
end
targets = zeros(Int, n)
for j in 1:c
    targets[nsamples*(j-1)+1:nsamples*j] .= j
end

# visualize the samples
ccp = palette(:tab10)
colors = [ccp[1] ccp[2] ccp[3] ccp[4]]
xtmp = 1:n
ytmp = zeros(n)
for j in 1:c
    ytmp[nsamples*(j-1)+1:nsamples*j] .= 0.2 * (j-2)
end
fig1 = scatter(xtmp, ytmp; group=targets, ylims=(-0.5,0.5), yticks=nothing, color=colors, legend=:outerright, xlabel="observation #", ylabel="True Classes")

# create a problem instance
problem = MVDAProblem(targets, X, intercept=true)

# set hyperparameters; note sparsity = 0.0 implies no model selection
ϵ = 0.5 * sqrt(2*c/(c-1))
sparsity = 0.0

# compute an initial solution w/ regularization
fill!(problem.coeff.all, 1/(p+1))
fit_regMVDA(MMSVD(), problem, ϵ, 1.0, gtol=1e-8)

# fit the model
@time fit_MVDA(MMSVD(), problem, ϵ, sparsity, verbose=true, nouter=1000, ninner=10^6, rtol=0.0, gtol=1e-12, dtol=1e-6, nesterov_threshold=100);
Y = problem.Y
B = problem.proj.all
V = unique(Y, dims=1)

# assign vertices and labels
Yfit = problem.X*B
labels = MVDA.classify(problem, problem.X)

# assess misclassification error
train_error = round(100 * (1 - sum(labels .== targets)/n), sigdigits=4)

# get distances and plot with assigned labels
dist = zeros(n, c)
fig2 = plot(xlabel="observation #", ylabel="distance", ylims=(0.0, 2.25), title="Misclassification Error: $(train_error) %", legend=nothing)
for j in 1:c
    dist[:,j] .= [ norm(Yfit[i,:] - V[j,:]) for i in 1:n]
    scatter!(fig2, xtmp, dist[:,j], label=j, color=colors[j])
end
hline!(fig2, [ϵ], label="", lw=3, style=:dash, color=:black)

# visualize data in vertex space
vertex_labels = map(v -> problem.vertex2label[v], eachrow(V))
fig3 = scatter(V[:,1], V[:,2], xlabel="y₁", ylabel="y₂", title="Vertex Space", group=vertex_labels, color=colors, marker=:star, markersize=8, aspect_ratio=1)
scatter!(Yfit[:,1], Yfit[:,2], group=targets, label="", color=colors, legend=:bottomright)
foreach(v -> draw_circle!(v, ϵ), problem.vertex)
fig3

layout = @layout [[a{0.2h}; b{0.8h}] c{0.6w}]
w, h = default(:size)
fig = plot(fig1, fig2, fig3, layout=layout, size=(2*w,1.1*h), margins=5Plots.mm)

savefig(fig, "~/Desktop/simulated2.png")
