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

function synthetic(a,b,c,d,e; rng::AbstractRNG=StableRNG(2000), prob::Real=1.0, m::Int=10^3, n::Int=500)
    if n < 2
        error("Number of features (n=$(n)) should be greater than 2.")
    end
    if prob < 0 || prob > 1
        error("Probability (prob=$(prob)) must satisfy 0 ≤ prob ≤ 1.")
    end

    # covariance matrix
    Σ = Matrix{Float64}(c*I, n, n)
    for j in 1:n, i in j+1:n
        Σ[j,i] = e
    end
    Σ[1,1] = a
    Σ[2,2] = b

    # Cluster A
    Σ1 = copy(Σ)
    Σ1[1,2] = +d

    # Cluster B
    Σ2 = copy(Σ)
    Σ2[1,2] = -d

    if !isposdef(Symmetric(Σ1)) || !isposdef(Symmetric(Σ2))
        error("At least one covariance matrix is not positive definite. Try decreasing parameters d or e relative to a, b, and c.")
    end

    # Simulate instances.
    X = Matrix{Float64}(undef, m, n)
    L1, _ = cholesky(Symmetric(Σ1))
    L2, _ = cholesky(Symmetric(Σ2))
    cluster = Vector{String}(undef, m)
    for i in axes(X, 1)
        # Sample features from Class A
        if rand(rng) > 0.5
            @views X[i, :] .= L1*randn(rng, n)
            cluster[i] = "A"
        else # Class B
            @views X[i, :] .= L2*randn(rng, n)
            cluster[i] = "B"
        end
    end
    
    StatsBase.transform!(StatsBase.fit(ZScoreTransform, X, dims=1), X)

    # Set coefficients
    beta = zeros(n)
    beta[1] = 10.0
    beta[2] = -10.0

    # Assign labels.
    y, L = Vector{Float64}(undef, m), Vector{String}(undef, m)
    inversions = 0
    for i in eachindex(y)
        xi = view(X, i, :)
        yi = sign(xi'*beta)
        if rand(rng) < prob
            y[i], L[i] = yi, ifelse(yi == 1, "A", "B")
        else
            y[i], L[i] = ifelse(cluster[i] == "A", 1, -1), cluster[i]
            inversions += 1
        end
    end
    @info """
    [ synthetic: $(m) instances / $(n) features ]"
        ∘ Pr(y | x) = $(prob)
        ∘ $inversions class inversions ($(inversions/m) Bayes error)
    """
    return L, X
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
