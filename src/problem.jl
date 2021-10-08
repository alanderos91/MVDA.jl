const Coefficients{T1,T2} = NamedTuple{(:all,:dim), Tuple{T1,T2}}
const Residuals{T1,T2,T3} = NamedTuple{(:main,:dist,:weighted), Tuple{T1,T2,T3}}

"""
```
__allocate_coeff__(T, nfeatures, nclasses)
```

Allocate a `NamedTuple` storing data similar in shape to model coefficients.
"""
function __allocate_coeff__(T, p, c)
    arr = similar(Matrix{T}, p, c-1)
    return (
        all=arr,                                # full coefficients vector
        dim=[view(arr, :, j) for j in 1:c-1]  # views into parameters for each class
    )
end

"""
```
__allocate_res__(T, nsamples, nfeatures, nclasses)
```

Allocate a `NamedTuple` storing data similar in shape to model residuals.
"""
function __allocate_res__(T, n, p, c)
    arr_main = similar(Matrix{T}, n, c-1)
    arr_dist = similar(Matrix{T}, p, c-1)
    arr_weighted = similar(Matrix{T}, n, c-1)
    
    # views into main residuals, R = Y - BX
    main = (
        all=arr_main,
        dim=[view(arr_main, :, j) for j in 1:c-1],
        sample=[view(arr_main, i, :) for i in 1:n],
    )

    # views into distance residuals, P - B
    dist = (
        all=arr_dist,
        dim=[view(arr_dist, :, j) for j in 1:c-1]
    )

    # views into weighted residuals, Z - BX
    weighted = (
        all=arr_weighted,
        dim=[view(arr_weighted, :, j) for j in 1:c-1],
        sample=[view(arr_weighted, i, :) for i in 1:n],
    )

    return (main=main, dist=dist, weighted=weighted,)
end

struct MVDAProblem{T,matT,labelT,viewT,R1,R2,R3}
    ##### data #####
    Y::matT
    X::matT
    vertex::Vector{Vector{T}}
    label2vertex::Dict{labelT,Vector{T}}
    vertex2label::Dict{Vector{T},labelT}
    intercept::Bool

    ##### model #####
    coeff::Coefficients{matT,viewT}
    coeff_prev::Coefficients{matT,viewT}
    
    ##### quadratic surrogate #####
    proj::Coefficients{matT,viewT}
    res::Residuals{R1,R2,R3}
    grad::Coefficients{matT,viewT}
end

function MVDAProblem{T}(Y, X, vertex, label2vertex, vertex2label, intercept, coeff, coeff_prev, proj, res, grad) where T <: Real
    # get type parameters
    matT = typeof(Y)
    labelT = keytype(label2vertex)
    viewT = typeof(coeff.dim)
    R1 = typeof(res.main)
    R2 = typeof(res.dist)
    R3 = typeof(res.weighted)
    
    MVDAProblem{T,matT,labelT,viewT,R1,R2,R3}(
        Y, X, vertex, label2vertex, vertex2label, intercept,
        coeff, coeff_prev,
        proj, res, grad,
    )
end

function MVDAProblem(labels, X; intercept=true)
    # modify data in case intercept is used
    if intercept
        X = [X ones(size(X, 1))]
    else
        X = copy(X)
    end

    # get problem info
    class = unique(labels)
    (n, p), c = size(X), length(class)
    T = Float64 # TODO

    # encode labels into vertices
    a = ( 1 + sqrt(c) ) / ( (c-1)^(3/2) )
    b = sqrt( c / (c-1) )
    vertex = Vector{Vector{T}}(undef, c)

    vertex[1] = 1 / sqrt(c-1) * ones(c-1)
    for j in 2:c
        v = -a * ones(c-1)
        v[j-1] += b
        vertex[j] = v
    end

    # assign classes to vertices and create response matrix
    label2vertex = Dict(class_j => vertex[j] for (j, class_j) in enumerate(class))
    vertex2label = Dict(vertex[j] => class_j for (j, class_j) in enumerate(class))
    Y = Matrix{T}(undef, n, c-1)
    for (i, label_i) in enumerate(labels)
        Y[i,:] .= label2vertex[label_i]
    end

    # allocate data structures for coefficients, projections, residuals, and gradient
    coeff = __allocate_coeff__(T, p, c)
    coeff_prev = __allocate_coeff__(T, p, c)
    proj = __allocate_coeff__(T, p, c)
    res = __allocate_res__(T, n, p, c)
    grad = __allocate_coeff__(T, p, c)

    return MVDAProblem{T}(
        Y, X, vertex, label2vertex, vertex2label, intercept,
        coeff, coeff_prev,
        proj, res, grad,
    )
end

"""
Returns the floating point type used for model coefficients.
"""
floattype(::MVDAProblem{T}) where T = T

"""
Return the design matrix, `X`, in the linear model `Y ∼ X * B`.
"""
get_design_matrix(problem::MVDAProblem) = problem.X

"""
Returns the number of samples, number of features, and number of categories, respectively.
"""
probdims(problem::MVDAProblem) = size(problem.X, 1), size(problem.X, 2) - problem.intercept, length(problem.vertex)

predict(problem::MVDAProblem, x::AbstractVector) = problem.proj.all' * x
predict(problem::MVDAProblem, X::AbstractMatrix) = map(xᵢ -> predict(problem, xᵢ), eachrow(X))

function classify(problem::MVDAProblem, x::AbstractVector)
    y = predict(problem, x)
    v = problem.vertex
    distances = [norm(y - v[j]) for j in eachindex(v)]
    j = argmin(distances)
    return problem.vertex2label[v[j]]
end

classify(problem::MVDAProblem, X::AbstractMatrix) = map(xᵢ -> classify(problem, xᵢ), eachrow(X))

function Base.show(io::IO, problem::MVDAProblem)
    n, p, c = probdims(problem)
    T = floattype(problem)
    respT = eltype(problem.Y)
    matT = eltype(problem.X)
    labelT = keytype(problem.label2vertex)
    print(io, "MVDAProblem{$(T)}")
    print(io, "\n  ∘ $(n) sample(s) ($(respT))")
    print(io, "\n  ∘ $(p) feature(s) ($(matT))")
    print(io, "\n  ∘ $(c) categories ($(labelT))")
    print(io, "\n  ∘ intercept? $(problem.intercept)")
end

struct NonLinearMVDAProblem{T,KERNEL,matT,labelT,viewT,R1,R2,R3}
    ##### data #####
    Y::matT
    X::matT
    K::matT
    vertex::Vector{Vector{T}}
    label2vertex::Dict{labelT,Vector{T}}
    vertex2label::Dict{Vector{T},labelT}
    intercept::Bool

    ##### model #####
    κ::KERNEL
    coeff::Coefficients{matT,viewT}
    coeff_prev::Coefficients{matT,viewT}
    
    ##### quadratic surrogate #####
    proj::Coefficients{matT,viewT}
    res::Residuals{R1,R2,R3}
    grad::Coefficients{matT,viewT}
end

function NonLinearMVDAProblem{T}(Y, X, K, vertex, label2vertex, vertex2label, intercept, κ, coeff, coeff_prev, proj, res, grad) where T <: Real
    # get type parameters
    KERNEL = typeof(κ)
    matT = typeof(Y)
    labelT = keytype(label2vertex)
    viewT = typeof(coeff.dim)
    R1 = typeof(res.main)
    R2 = typeof(res.dist)
    R3 = typeof(res.weighted)
    
    NonLinearMVDAProblem{T,KERNEL,matT,labelT,viewT,R1,R2,R3}(
        Y, X, K, vertex, label2vertex, vertex2label, intercept,
        κ, coeff, coeff_prev,
        proj, res, grad,
    )
end

function NonLinearMVDAProblem(κ::Kernel, labels, X; intercept=true)
    # modify data in case intercept is used
    K = kernelmatrix(κ, X, obsdim=1)
    if intercept
        K = [K ones(size(K, 1))]
    end

    # get problem info
    class = unique(labels)
    (n, p), c = size(X), length(class)
    T = Float64 # TODO

    # encode labels into vertices
    a = ( 1 + sqrt(c) ) / ( (c-1)^(3/2) )
    b = sqrt( c / (c-1) )
    vertex = Vector{Vector{T}}(undef, c)

    vertex[1] = 1 / sqrt(c-1) * ones(c-1)
    for j in 2:c
        v = -a * ones(c-1)
        v[j-1] += b
        vertex[j] = v
    end

    # assign classes to vertices and create response matrix
    label2vertex = Dict(class_j => vertex[j] for (j, class_j) in enumerate(class))
    vertex2label = Dict(vertex[j] => class_j for (j, class_j) in enumerate(class))
    Y = Matrix{T}(undef, n, c-1)
    for (i, label_i) in enumerate(labels)
        Y[i,:] .= label2vertex[label_i]
    end

    # allocate data structures for coefficients, projections, residuals, and gradient
    coeff = __allocate_coeff__(T, n+intercept, c)
    coeff_prev = __allocate_coeff__(T, n+intercept, c)
    proj = __allocate_coeff__(T, n+intercept, c)
    res = __allocate_res__(T, n, n+intercept, c)
    grad = __allocate_coeff__(T, n+intercept, c)

    return NonLinearMVDAProblem{T}(
        Y, X, K, vertex, label2vertex, vertex2label, intercept,
        κ, coeff, coeff_prev,
        proj, res, grad,
    )
end

"""
Returns the floating point type used for model coefficients.
"""
floattype(::NonLinearMVDAProblem{T}) where T = T

"""
Return the design matrix, `K`, in the linear model `Y ∼ K * Γ`.
"""
get_design_matrix(problem::NonLinearMVDAProblem) = problem.K

"""
Returns the number of samples, number of features, and number of categories, respectively.
"""
probdims(problem::NonLinearMVDAProblem) = size(problem.X, 1), size(problem.X, 2), length(problem.vertex)

function predict(problem::NonLinearMVDAProblem, x::AbstractVector)
    n, _, c = probdims(problem)
    κ = problem.κ
    Γ = problem.coeff.all
    ϕ = zeros(c-1)

    for j in eachindex(ϕ)
        for (i, xᵢ) in enumerate(eachrow(problem.X))
            ϕ[j] += Γ[i,j] * κ(xᵢ, x)
        end
        ϕ[j] += ifelse(problem.intercept, Γ[end,j], zero(ϕ[j]))
    end

    return ϕ
end

predict(problem::NonLinearMVDAProblem, X::AbstractMatrix) = map(xᵢ -> predict(problem, xᵢ), eachrow(X))

function classify(problem::NonLinearMVDAProblem, x::AbstractVector)
    y = predict(problem, x)
    v = problem.vertex
    distances = [norm(y - v[j]) for j in eachindex(v)]
    j = argmin(distances)
    return problem.vertex2label[v[j]]
end

classify(problem::NonLinearMVDAProblem, X::AbstractMatrix) = map(xᵢ -> classify(problem, xᵢ), eachrow(X))

function Base.show(io::IO, problem::NonLinearMVDAProblem)
    n, p, c = probdims(problem)
    T = floattype(problem)
    respT = eltype(problem.Y)
    matT = eltype(problem.X)
    labelT = keytype(problem.label2vertex)
    print(io, "NonLinearMVDAProblem{$(T)}")
    print(io, "\n  ∘ $(typeof(problem.κ))")
    print(io, "\n  ∘ $(n) sample(s) ($(respT))")
    print(io, "\n  ∘ $(p) feature(s) ($(matT))")
    print(io, "\n  ∘ $(c) categories ($(labelT))")
    print(io, "\n  ∘ intercept? $(problem.intercept)")
end
