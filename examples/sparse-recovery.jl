using MVDA, Random, LinearAlgebra, StableRNGs
using StatsBase, Statistics
using ProgressMeter

# Compute an initial solution w/ regularization. Take this estimate as the ground truth.
function __init__(λ, problem, ϵ, gtol)
    MVDA.init!(MMSVD(), problem, ϵ, λ, verbose=false, maxiter=10^6, nesterov_threshold=0, gtol=gtol)
    B0 = copy(problem.coeff.all)
    return B0
end

# Solve for s-sparse solution.
function __solve__(s, problem, ϵ, gtol, dtol)
    MVDA.fit(MMSVD(), problem, ϵ, s, verbose=false, ninner=10^6, nouter=200, nesterov_threshold=0, gtol=gtol, dtol=dtol, rtol=0.0)
    B = problem.proj.all
    return B
end

# Check classification accuracy against X.
function __accuracy__(B, problem, targets, X)
    n = problem.n
    copyto!(problem.proj.all, B)
    labels = MVDA.classify(problem, X)
    return round(100 * (1 - sum(labels .== targets) / n), sigdigits=4)
end

# Check MSE against ground truth B0.
mse(B, B0) = mean( (B - B0) .^ 2 ) / dot(B0, B0)

# Check contingency table by comparing estimate x against ground truth y.
function discovery_metrics(x, y)
    TP = FP = TN = FN = 0
    for (xi, yi) in zip(eachrow(x), eachrow(y))
        nxi, nyi = norm(xi), norm(yi)
        TP += (nxi != 0) && (nyi != 0)
        FP += (nxi != 0) && (nyi == 0)
        TN += (nxi == 0) && (nyi == 0)
        FN += (nxi == 0) && (nyi != 0)
    end
    return (TP, FP, TN, FN)
end

function run(filename, λ)
    # Fix problem parameters.
    c = 10
    p = 50
    rng = StableRNG(1903)

    # Define range of values for various simulation settings.
    separation_levels = (1.0, 3.0)
    noise_levels = (1e0, 5e0, 1e1)
    dataset_sizes = (10, 40, 100)
    sparsity_levels = Float64[1 - k/p for k in p:-1:0]

    # Allocate arrays for interesting metrics.
    m = length(noise_levels)
    train_error = zeros(p+1, m)
    mean_squared_error = zeros(p+1, m)

    @info "Initializing ouptut file: $(filename).dat"
    open("$(filename).dat", "w+") do io
        write(io, "n,p,c,nclass,ncausal,separation,sigma,sparsity,error,MSE,TP,FP,TN,FN\n")
    end

    for samples_per_class in dataset_sizes, separation in separation_levels
        for (j, sigma) in enumerate(noise_levels)
            # Simulate ground truth with the given parameters.
            n = samples_per_class*c
            targets, X, B0 = MVDA.simulate_ground_truth(p, c, samples_per_class, d=separation, rng=rng, sigma=sigma)

            # Standardize ALL data based on the training set.
            F = StatsBase.fit(ZScoreTransform, X, dims=1)
            has_nan = any(isnan, F.scale) || any(isnan, F.mean)
            has_inf = any(isinf, F.scale) || any(isinf, F.mean)
            has_zero = any(iszero, F.scale)
            if has_nan
                error("Detected NaN in z-score.")
            elseif has_inf
                error("Detected Inf in z-score.")
            elseif has_zero
                for idx in eachindex(F.scale)
                    x = F.scale[idx]
                    F.scale[idx] = ifelse(iszero(x), one(x), x)
                end
            end
            StatsBase.transform!(F, X)

            # Figure out how to group samples together and create a problem instance.
            sortidx = sortperm(targets)
            problem = MVDAProblem(targets, X, intercept=false)

            # Fix hyperparameter controlling distance insensitivity.
            ϵ = MVDA.maximal_deadzone(problem)

            # Create closures specific to our problem instance.
            make_data_plot() = __make_data_plot__(targets, n, c, samples_per_class, ccp, colors)
            make_distance_plot(B) = __make_distance_plot__(B, problem, targets, ϵ, sortidx, ccp, colors)
            init() = __init__(λ, problem, ϵ, 1e-6)
            solve(s) = __solve__(s, problem, ϵ, 1e-3, 1e-3)
            accuracy(B) = __accuracy__(B, problem, targets, X)

            @showprogress "$(samples_per_class) per class / d = $(separation) / σ = $(sigma)... " for (i, s) in enumerate(sparsity_levels)
                # Estimate s-sparse coefficients.
                if s != 0
                    B = solve(s)
                else
                    B = init()
                end

                train_error[i,j] = accuracy(B)
                mean_squared_error[i,j] = mse(B, B0)
                TP, FP, TN, FN = discovery_metrics(B[1:p,:], B0[1:p,:])

                open("$(filename).dat", "a+") do io
                    write(io, "$(n),$(p),$(c),$(samples_per_class),$(c),$(separation),$(sigma),$(s),$(train_error[i,j]),$(mean_squared_error[i,j]),$(TP),$(FP),$(TN),$(FN)\n")
                end
            end
        end
    end
end

# Parse input arguments and run.
length(ARGS) < 2 && error("Script requires a filename (no extension required) and a value λ > 0 for regularization.")
filename = ARGS[1]
λ = parse(Float64, ARGS[2])
@info "Output file: $(filename)"
@info "λ value: $(λ)"
run(filename, λ)
