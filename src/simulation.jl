"""
Simulate `n` points in 2D space distributed over `c` nested circles.

# Keyword Arguments

- `rng=StableRNG(1234)`: A random number generator for reproducibility.
- `p=8//10`: A prior probability on how well a sample's position predicts its class (Bayes error).
"""
function nested_circles(n::Int, c::Int=2; p::Real=8//10, rng::AbstractRNG=StableRNG(1234))
    # allocate outputs
    X = Matrix{Float64}(undef, n, 2)
    class = Vector{Int}(undef, n)

    # for each sample...
    for i in axes(X, 1)
        # generate a point (x, y) via polar coordinates
        r = sqrt(c * rand(rng))
        θ = 2*π * rand(rng)
        x = r * cos(θ)
        y = r * sin(θ)

        X[i, 1] = x
        X[i, 2] = y

        # assign a class via an intermediate point and the Bayes error, p
        class_bayes = ceil(Int, x^2 + y^2)

        if rand(rng) ≤ p # accept proposed class
            class[i] = class_bayes
        else # assign a random class uniformly over all remaining classes
            class[i] = rand(rng, setdiff(1:c, class_bayes))
        end
    end

    return class, X
end

"""
Simulate `n` waveform data points with `p` features, over 3 classes.

# Keyword Arguments

- `rng=StableRNG(1234)`: A random number generator for reproducibility.
- `m=6`: Waveform's peak.
- `r=4`: Shift parameter.
- `s=11`: Shift parameter.
"""
function waveform(n::Int, p::Int; m::Int=6, r::Int=4, s::Int=11, rng::AbstractRNG=StableRNG(1234))
    # allocate outputs
    X = Matrix{Float64}(undef, n, p)
    class = Vector{Int}(undef, n)
    
    # for each sample...
    for i in axes(X, 1)
        u₁ = rand(rng)
        u₂ = rand(rng)
        h₁(t) = max(m - abs(t-s), 0)
        h₂(t) = h₁(t-r)
        h₃(t) = h₁(t+r)
        if u₁ ≤ 1//3
            class[i] = 1
            for j in axes(X, 2)
                ϵ = randn(rng)
                X[i,j] = u₂*h₁(j) + (1-u₂)*h₂(j) + ϵ
            end
        elseif 1//3 < u₁ ≤ 2//3
            class[i] = 2
            for j in axes(X, 2)
                ϵ = randn(rng)
                X[i,j] = u₂*h₁(j) + (1-u₂)*h₃(j) + ϵ
            end
        else
            class[i] = 3
            for j in axes(X, 2)
                ϵ = randn(rng)
                X[i,j] = u₂*h₂(j) + (1-u₂)*h₃(j) + ϵ
            end
        end
    end

    return class, X
end

"""
Simulate `p`-dimensional feature vectors equally distributed over `c` classes with known effect sizes.

Only a subset of features are informative in predicting a sample's class. Both degree of class separation `d` and noise level `sigma` affect class masking.

# Keyword Arguments

- `rng=StableRNG(1234)`: A random number generator for reproducibility.
- `d=1.0`: Degree of class separation, with larger values correspoding to well-separated classes.
- `sigma=1.0`: Standard deviation of Normal distribution used to seed features.
- `ncausal=2`: The number of informative features.
"""
function masked_classes(p::Int, c::Int, samples_per_class::Int;
    rng::AbstractRNG=StableRNG(1234),
    d::Real=1.0,
    sigma::Real=1.0)
    # Sanity checks
    any(≤(0), (p, c, samples_per_class)) && error("Values for `n`, `p`, and `samples_per_class` must be positive.")
    d < 0 && error("Class separation `d` must be a nonnegative value.")
    sigma ≤ 0 && error("Noise level `sigma` must be a positive value.")

    A1 = ( 1 + sqrt(c) ) / ( (c-1)^(3/2) )
    A2 = sqrt( c / (c-1) )
    vertex = Vector{Vector{Float64}}(undef, c)
    vertex[1] = 1 / sqrt(c-1) * ones(c-1)
    for j in 2:c
        v = -A1 * ones(c-1)
        v[j-1] += A2
        vertex[j] = v
    end

    # Initialize with iid Normal deviate samples, N(0, σ²)
    n = c * samples_per_class
    X = 1/sigma * randn(rng, n, p)
    B = zeros(p, c-1)
    for i in 1:c
        idx = samples_per_class*(i-1)+1:samples_per_class*i
        θ = π/3 + 2*(i-1)*π/c
        a = iseven(i) ? cos(θ) : sin(θ)
        X[idx,i] .= d * sign(a) .+ a * sigma * randn(rng, samples_per_class)
        B[i,:] .= sign(a) * vertex[i] / c
    end

    # Assign classes and shuffle samples around.
    Y = X*B
    targets = map(y -> argmin([max(0, norm(y-v)) for v in vertex]), eachrow(Y))

    idx = randperm(rng, n)
    # targets, X = targets[idx], X[idx,:]
    return targets, X, B
end

"""
Simulate `p`-dimensional feature vectors equally distributed over `c` based on a recipe from Wang & Shen 2007.

Only two predictors are informative. Both degree of class separation `d` and noise level `sigma` affect class masking.

# Keyword Arguments

- `rng=StableRNG(1234)`: A random number generator for reproducibility.
- `d=1.0`: Degree of class separation, with larger values correspoding to well-separated classes.
- `sigma=1.0`: Standard deviation of Normal distribution used to seed features.
"""
function WS2007(p::Int, c::Int, samples_per_class::Int;
    rng::AbstractRNG=StableRNG(1234),
    d::Real=1.0,
    sigma::Real=1.0,)
    # Sanity checks
    any(≤(0), (p, c, samples_per_class)) && error("Values for `n`, `p`, and `samples_per_class` must be positive.")
    d < 0 && error("Class separation `d` must be a nonnegative value.")
    sigma ≤ 0 && error("Noise level `sigma` must be a positive value.")

    # Initialize with iid Normal deviate samples, N(0, σ²)
    n = c * samples_per_class
    targets = zeros(Int, n)
    X = sigma * randn(rng, n, p)
    for i in 1:c
        idx = samples_per_class*(i-1)+1:samples_per_class*i
        
        # Causal predictors
        θ = π/3 + 2*(i-1)*π/c
        X[idx,1] .= d*cos(θ)
        X[idx,2] .= d*sin(θ)
        
        # Class assignment
        targets[idx] .= i
    end

    # Shuffle samples around.
    idx = randperm(rng, n)
    targets, X = targets[idx], X[idx,:]
    return targets, X
end

"""
Simulate Gaussian point data with `n` samples and `c` classes.

# Keyword Arguments

- `rng=StableRNG(1234)`: A random number generator for reproducibility.
- `sigma=1.0`: Standard deviation for Normal distribution.
"""
function gaussian_clouds(n::Int, c::Int=3; sigma::Real=1.0, rng::AbstractRNG=StableRNG(1234))
    # Define one half of cluster centers.
    centers = [(k, (k-1)*π / c) for k in 1:c]

    # Initialize data matrix with iid N(0,1) variates
    target, X = Vector{Int}(undef, n), randn(rng, n, 2)
    σ = sigma

    for i in axes(X, 1)
        # Sample a center uniformly from the three classes and randomly reflect it.
        (class, θ) = rand(rng, centers)
        rand(rng, (true, false)) && (θ += π)

        X[i,1] = cos(θ) + σ * X[i,1]
        X[i,2] = sin(θ) + σ * X[i,2]
        target[i] = class
    end

    return target, X
end

#
#   Correlation matrices
#
"""
    cor_toeplitz(p::Integer, rho::Real)

Generate a correlation matrix with Toeplitz structure, `Σ[i,j] = rho^abs(i-j)`.
"""
function cor_toeplitz(p::Integer, rho::Real)
    Σ = zeros(p, p)
    for j in 1:p, i in j:p
        Σ[i,j] = rho^abs(i-j)
    end

    if !isposdef(Symmetric(Σ, :L))
        error("Toeplitz correlation matrix may not be positive definite.")
    end

    return Symmetric(Σ, :L)
end

#
#   Multivariate Normal deviates
#
"""
    rand_mvn_normal!(rng::AbstractRNG, x::AbstractVector, Σ::AbstractMatrix)

Fill `x` with 0-mean multivariate normal deviates, given the covariance structure `Σ`.
"""
function rand_mvn_normal!(rng::AbstractRNG, x::AbstractVector, Σ::AbstractMatrix)
    rand_mvn_normal!(rng, x, cholesky(Σ))
end

"""
    rand_mvn_normal!(rng::AbstractRNG, x::AbstractVector, F::Cholesky)

Fill `x` with 0-mean multivariate normal deviates, given the covariance structure `F.L*F.U`.
"""
function rand_mvn_normal!(rng::AbstractRNG, x::AbstractVector, F::Cholesky)
    randn!(rng, x)
    lmul!(F.L, x)
    return x
end

"""
    rand_mvn_normal!(rng::AbstractRNG, x::AbstractVector, arr::AbstractVector)

Fill `x` with 0-mean multivariate normal deviates, given the block-diagonal covariance structure specified by components of `arr` (either `AbstractMatrix` or `Cholesky`).
"""
function rand_mvn_normal!(rng::AbstractRNG, x::AbstractVector, arr::AbstractVector)
    a = 1
    for F in arr
        m = size(F, 1)
        b = a + m - 1
        rand_mvn_normal!(rng, view(x, a:b), F)
        a = b + 1
    end
    return x
end

"""
    rand_mvn_normal!(rng::AbstractRNG, X::AbstractMatrix, F)

Fill rows of `X` with 0-mean multivariate normal deviates, given the covariance structure
implied by `F`.
"""
function rand_mvn_normal!(rng::AbstractRNG, X::AbstractMatrix, F)
    n = size(X, 1)
    for i in 1:n
        rand_mvn_normal!(rng, view(X, i, :), F)
    end
    return X
end

#
#   Covariates with known correlation structure
#
function synth_covariates(rng::AbstractRNG, n::Integer, p::Integer, Σ)
    X = rand_mvn_normal!(rng, Matrix{Float64}(undef, n, p), Σ)
    # StatsBase.transform!(
    #     StatsBase.fit(ZScoreTransform, X, dims=1),
    #     X
    # )
    return X
end

#
#   Structured coefficient matrices for VDA models
#
function synth_coef!(rng::AbstractRNG, b::AbstractVector, k::Integer)
    for j in 1:k
        b[j] = rand(rng, (+1, -1))
    end
    return b
end

function synth_coef_hom(rng::AbstractRNG, p::Integer, c::Integer, k::Integer)
    B = zeros(p, c)
    idx = randperm(rng, p)[1:k]
    for l in 1:c
        synth_coef!(rng, view(B, idx, l), k)
    end
    return B, idx
end

function synth_coef_het(rng::AbstractRNG, p::Integer, c::Integer, k::Integer)
    B = zeros(p, c)
    idx = [randperm(rng, p)[1:k] for _ in 1:c]
    for l in 1:c
        synth_coef!(rng, view(B, idx[l], l), k[l])
    end
    return B, idx
end

function synth_coef_het(rng::AbstractRNG, p::Integer, c::Integer, k::AbstractVector)
    B = zeros(p, c)
    idx = [randperm(rng, p)[1:k[l]] for l in 1:c]
    for l in 1:c
        synth_coef!(rng, view(B, idx[l], l), k[l])
    end
    return B, idx
end

#
#   Class assignment with VDA model
#
function synth_assignment(X::AbstractMatrix, B::AbstractMatrix)
    # dimensions
    n = size(X, 1)
    c = size(B, 2)

    # vertex encoding
    enc = StandardSimplexEncoding(c)
    V = enc.vertex

    # assign classes based on VDA rule
    Y = X*B
    class = Vector{String}(undef, n)
    for i in 1:n
        yi = view(Y, i, :)
        l = argmin(norm(yi - V[l]) for l in 1:c)
        class[i] = "Class $(l)"
        yi .= V[l]
    end

    return class, Y
end

#
#   Synthetic data with feature-homogeneous classes
#
function vdasynth1(n::Integer, p::Integer, c::Integer, k::Integer;
    rng::AbstractRNG=Random.default_rng(),
    rho::Real=0.5,
    SNR::Real=1.0,
)
    # sanity checks
    if n < 2
        error("Number of samples (n=$(n)) should be greater than 1.")
    end
    if k > p
        error("Number of true predictors (k=$(k)) should be less than total predictors (p=$(p)).")
    end
    if SNR <= 0
        error("SNR must be given as a positive real number.")
    end

    Σ = cor_toeplitz(p, rho)
    X = synth_covariates(rng, n, p, cholesky!(Σ))
    B, idx = synth_coef_hom(rng, p, c, k)
    class, Y = synth_assignment(X, B)

    # shift X to make B closer to the 'true' model wrt VDA
    sigma = sqrt(mean(var(view(Y, :, l)) for l in axes(Y, 2)) / SNR)
    D = (view(B, idx, :)' \ (Y - X*B - sigma*randn(rng, n, c))')'
    X[:, idx] .= view(X, :, idx) + D

    @info """
    [ VDA Synthetic 1: $(n) instances / $(p) features ]
        ∘ $(k) true features
        ∘ Homogeneous classes
        ∘ Toeplitz correlation structure, ρ = $(rho)
    """
    return class, Y, X, B
end

#
#   Synthetic data with feature-heterogeneous classes
#
function vdasynth2(n::Integer, p::Integer, c::Integer, k::Vector{T};
    rng::AbstractRNG=StableRNG(2000),
    rho::Vector{F}=[0.5],
    SNR::Real=1.0,
) where {T <: Integer, F <: Real}
    # sanity checks
    if n < 2
        error("Number of samples (n=$(n)) should be greater than 1.")
    end
    if any(>(p), k)
        error("Number of true predictors (k=$(k)) should be less than total predictors (p=$(p)).")
    end
    if SNR <= 0
        error("SNR must be given as a positive real number.")
    end

    pg, r = divrem(p, length(rho))
    Σ = [cor_toeplitz(pg + (g == 1) * r, rho[g]) for g in eachindex(rho)]
    X = synth_covariates(rng, n, p, cholesky!.(Σ))
    B, idx = synth_coef_het(rng, p, c, k)
    class, Y = synth_assignment(X, B)

    # shift X to make B closer to the 'true' model wrt VDA
    sup = sort!(union(idx...))
    sigma = sqrt(mean(var(view(Y, :, l)) for l in axes(Y, 2)) / SNR)
    D = (view(B, sup, :)' \ (Y - X*B - sigma*randn(rng, n, c))')'
    X[:, sup] .= view(X, :, sup) + D
    
    @info """
    [ VDA Synthetic 2: $(n) instances / $(p) features ]
        ∘ $(join(k, ", ", " and ")) true features
        ∘ Heterogeneous classes
        ∘ Toeplitz correlation structure, ρ = $(rho)
    """
    return class, Y, X, B
end

function spiral(class_sizes;
    rng::AbstractRNG=StableRNG(1903),
    max_radius::Real=7.0,
    x0::Real=-3.5,
    y0::Real=3.5,
    angle_start::Real=π/8,
    prob::Real=1.0,
)
    if length(class_sizes) != 3
        error("Must specify 3 classes (length(class_sizes)=$(length(class_sizes))).")
    end
    if max_radius <= 0
        error("Maximum radius (max_radius=$(max_radius)) must be > 0.")
    end
    if angle_start < 0
        error("Starting angle (angle_start=$(angle_start)) should satisfy 0 ≤ θ ≤ 2π.")
    end
    if prob < 0 || prob > 1
        error("Probability (prob=$(prob)) must satisfy 0 ≤ prob ≤ 1.")
    end

    # Extract parameters.
    N = sum(class_sizes)
    max_A, max_B, max_C = class_sizes

    # Simulate the data.
    L, X = Vector{String}(undef, N), Matrix{Float64}(undef, N, 2)
    x, y = view(X, :, 1), view(X, :, 2)
    inversions = 0
    for i in 1:N
        if i ≤ max_A
            (class, k, n, θ) = ("A", i, max_A, angle_start)
            noise = 0.1
        elseif i ≤ max_A + max_B
            (class, k, n, θ) = ("B", i-max_A+1, max_B, angle_start + 2π/3)
            noise = 0.2
        else
            (class, k, n, θ) = ("C", i-max_A-max_B+1, max_C, angle_start + 4π/3)
            noise = 0.3
        end

        # Compute coordinates.
        angle = θ + π * k / n
        radius = max_radius * (1 - k / (n + n / 5))

        x[i] = x0 + radius*cos(angle) + noise*randn(rng)
        y[i] = y0 + radius*sin(angle) + noise*randn(rng)
        if rand(rng) < prob
            L[i] = class
        else
            L[i] = rand(rng, setdiff(["A", "B", "C"], [class]))
            inversions += 1
        end
    end

    @info """
    [ spiral: $(N) instances / 2 features / 3 classes ]
        ∘ Pr(y | x) = $(prob)
        ∘ $inversions class inversions ($(inversions/N) Bayes error)
    """

    return L, X
end
