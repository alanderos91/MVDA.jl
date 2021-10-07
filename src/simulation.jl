function generate_nested_circle(n::Int, c::Int=2; p::Real=8//10, rng::AbstractRNG=StableRNG(1234))
    # allocate outputs
    X = Matrix{Float64}(undef, n, 2)
    class = Vector{Int}(undef, n)

    # for each sample...
    for i in axes(X, 1)
        # generate a point (x, y) via polar coordinates
        r = rand(rng) * c
        θ = 2*π * rand(rng)
        x = r * cos(θ)
        y = r * sin(θ)

        X[i, 1] = x
        X[i, 2] = y

        # assign a class via an intermediate point and the Bayes error, p
        class_bayes = ceil(Int, sqrt(x^2 + y^2))
        if rand(rng) ≤ p # accept proposed class
            class[i] = class_bayes
        else # assign a random class uniformly over all possible classes
            u = ceil(Int, rand(rng) * (c-1))
            class[i] = ifelse(u < class_bayes, u, u+1)
        end
    end

    return class, X
end

function generate_waveform(n::Int, p::Int; m::Int=6, r::Int=4, s::Int=11, rng::AbstractRNG=StableRNG(1234))
    # allocate outputs
    X = Matrix{Float64}(undef, n, p)
    class = Vector{Int}(undef, n)
    
    # for each sample...
    for i in axes(X, 1)
        u₁ = rand(rng)
        u₂ = rand(rng)

        if u₁ ≤ 1//3
            class[i] = 1
            for j in axes(X, 2)
                ϵ = randn(rng)
                A = max(m-abs(j-s), 0)
                B = max(m-abs(j-r-s), 0)
                X[i,j] = u₂*A + (1-u₂)*B + ϵ
            end
        elseif 1//3 < u₁ ≤ 2//3
            class[i] = 2
            for j in axes(X, 2)
                ϵ = randn(rng)
                A = max(m-abs(j-s), 0)
                B = max(m-abs(j+r-s), 0)
                X[i,j] = u₂*A + (1-u₂)*B + ϵ
            end
        else
            class[i] = 3
            for j in axes(X, 2)
                ϵ = randn(rng)
                A = max(m-abs(j-r-s), 0)
                B = max(m-abs(j+r-s), 0)
                X[i,j] = u₂*A + (1-u₂)*B + ϵ
            end
        end
    end

    return class, X
end
