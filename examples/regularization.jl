using MVDA, Plots, Statistics, Random, LinearAlgebra, LaTeXStrings, StableRNGs

font_main = Plots.font("Computer Modern", 12)
font_small = Plots.font("Computer Modern", 10)
w, h = default(:size)
upscale = 1.5
default(
    markerstrokewidth=0.5,
    titlefont=font_main,
    guidefont=font_main,
    tickfont=font_small,
    legendfont=font_small,
    size=(upscale*w, upscale*h),
    dpi=200
)

function draw_circle!(v, ϵ)
    xs = v[1] .+ range(-ϵ, ϵ, length=201)
    ys1 = sqrt.(ϵ^2 .- (xs .- v[1]) .^2) .+ v[2]
    ys2 = -sqrt.(ϵ^2 .- (xs .- v[1]) .^2) .+ v[2]
    plot!(xs, ys1, color=:black, style=:dash, label="")
    plot!(xs, ys2, color=:black, style=:dash, label="")
end

function run(dir, λs)
    # problem dimensions
    n = 300
    p = 1
    c = 3
    
    # simulate 3 classes with single predictor as in Wu, Lange 2010
    rng = StableRNG(1234)
    μ = Float64[-4,0,4]
    targets = [ones(Int, 100); 2*ones(Int, 100); 3*ones(Int,100)]
    X = zeros(n, p)
    for j in 1:c
        X[100*(j-1)+1:100*j,:] .= μ[j] .+ randn(rng, 100)
    end
    
    # visualize the samples
    ccp = palette(:seaborn_colorblind)
    colors = [ccp[1] ccp[2] ccp[3]]
    ytmp = [-0.2*ones(100); 0*ones(100); 0.2*ones(100)]
    fig1 = scatter(X, ytmp; group=targets, ylims=(-0.5,0.5), yticks=nothing, palette=ccp, legend=:outerright, xlabel=L"x_{1}", title="True Classes vs Feature Vectors")
    
    # create a problem instance
    problem = MVDAProblem(targets, X, intercept=true)
    
    # set hyperparameters
    ϵ = MVDA.maximal_deadzone(problem)

    for λ in λs
        # compute an initial solution w/ regularization
        @time MVDA.init!(SD(), problem, ϵ, λ, verbose=true, maxiter=10^4, nesterov_threshold=0, gtol=1e-6)

        Y = problem.Y
        B = problem.coeff.all
        V = hcat([problem.label2vertex[j] for j in 1:c]...)
        copyto!(problem.proj.all, B)

        # assign vertices and labels
        Yfit = problem.X*B
        labels = MVDA.classify(problem, X)

        # assess misclassification error
        train_error = round(100 * (1 - sum(labels .== targets)/n), sigdigits=4)

        # get distances and plot with assigned labels
        dist = zeros(n, c)
        fig2 = plot(xlabel=L"x_{1}", ylabel="Distance to Class Vertex", ylims=(0.0, 2.25), title="Misclassification Error: $(train_error) %", legend=nothing)
        for j in 1:c
            dist[:,j] .= [ norm(Yfit[i,:] - V[:,j]) for i in 1:n]
            scatter!(fig2, X, dist[:,j], label=j, palette=ccp, color=colors[j])
        end
        hline!(fig2, [ϵ], label="", lw=3, style=:dash, color=:black)

        # visualize data in vertex space
        vertex_labels = map(v -> problem.vertex2label[v], eachcol(V))
        fig3 = scatter(V[1,:], V[2,:], xlabel=L"y_{1}", ylabel=L"y_{2}", title="Classification in Vertex Space", group=vertex_labels, palette=ccp, legend=:bottomright, marker=:star, markersize=upscale*8, aspect_ratio=1)
        scatter!(Yfit[:,1], Yfit[:,2], label=nothing, palette=ccp, color=colors[labels])
        foreach(v -> draw_circle!(v, ϵ), problem.vertex)

        layout = @layout [[a{0.2h}; b{0.8h}] c{0.6w}]
        fig = plot(fig1, fig2, fig3, layout=layout, size=(2*w,1.1*h), margins=5Plots.mm)

        filename = joinpath(dir, "reg-lambda=$(λ).png")
        @info "Saving figure to $(filename)"
        savefig(fig, filename)
    end
end

# Parse input arguments and run.
length(ARGS) < 2 && error("Script requires a directory and at least one input λ > 0 for regularization.")
dir = ARGS[1]
λs = map(arg -> parse(Float64, arg), ARGS[2:end])
@info "Output directory: $(dir)"
@info "λ values: $(λs)"
run(dir, λs)
