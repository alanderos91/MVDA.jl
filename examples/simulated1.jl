using MVDA, Plots, Statistics, Random, LinearAlgebra

function draw_circle!(v, ϵ)
    xs = v[1] .+ range(-ϵ, ϵ, length=201)
    ys1 = sqrt.(ϵ^2 .- (xs .- v[1]) .^2) .+ v[2]
    ys2 = -sqrt.(ϵ^2 .- (xs .- v[1]) .^2) .+ v[2]
    plot!(xs, ys1, color=:black, style=:dash, label="")
    plot!(xs, ys2, color=:black, style=:dash, label="")
end

# problem dimensions
n = 300
p = 1
c = 3

# simulate 3 classes with single predictor as in Wu, Lange 2010
μ = Float64[-4,0,4]
targets = [ones(Int, 100); 2*ones(Int, 100); 3*ones(Int,100)]
X = zeros(n, p)
for j in 1:c
    X[100*(j-1)+1:100*j,:] .= μ[j] .+ randn(100)
end

# visualize the samples
ccp = palette(:tab10)
colors = [ccp[1] ccp[2] ccp[3]]
ytmp = [-0.2*ones(100); 0*ones(100); 0.2*ones(100)]
fig1 = scatter(X, ytmp; group=targets, ylims=(-0.5,0.5), yticks=nothing, color=colors, legend=:outerright, xlabel="xᵢ", title="True Classes")

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
B = problem.coeff.all
V = unique(Y, dims=1)

# assign vertices and labels
Yfit = problem.X*B
labels = MVDA.classify(problem, problem.X)

# assess misclassification error
train_error = round(100 * (1 - sum(labels .== targets)/n), sigdigits=4)

# get distances and plot with assigned labels
dist = zeros(n, c)
fig2 = plot(xlabel="xᵢ", ylabel="distance", ylims=(0.0, 2.25), title="Misclassification Error: $(train_error) %", legend=:bottomright)
for j in 1:c
    dist[:,j] .= [ norm(Yfit[i,:] - V[j,:]) for i in 1:n]
    scatter!(fig2, X, dist[:,j], label=j, color=colors[j])
end
hline!(fig2, [ϵ], label="", lw=3, style=:dash, color=:black)

# visualize data in vertex space
vertex_labels = map(v -> problem.vertex2label[v], eachrow(V))
fig3 = scatter(V[:,1], V[:,2], xlabel="y₁", ylabel="y₂", title="Vertex Space", group=vertex_labels, colors=colors, legend=nothing, marker=:star, markersize=8, aspect_ratio=1)
scatter!(Yfit[:,1], Yfit[:,2], group=labels, label="", color=colors, legend=:bottomright)
foreach(v -> draw_circle!(v, ϵ), problem.vertex)
fig3

layout = @layout [[a{0.2h}; b{0.8h}] c{0.6w}]
w, h = default(:size)
fig = plot(fig1, fig2, fig3, layout=layout, size=(2*w,1.1*h), margins=5Plots.mm)

savefig(fig, "~/Desktop/simulated1.png")
