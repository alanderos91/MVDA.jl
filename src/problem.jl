const Coefficients{T1,T2} = NamedTuple{(:all,:dim), Tuple{T1,T2}}
const Residuals{T1,T2,T3} = NamedTuple{(:main,:dist,:weighted), Tuple{T1,T2,T3}}

"""
```
__allocate_coeff__(T, nfeatures, nclasses)
```

Allocate a `NamedTuple` storing data similar in shape to model coefficients.
"""
function __allocate_coeff__(T, n, c)
    arr = similar(Matrix{T}, n, c-1)
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

    ##### model #####
    coeff::Coefficients{matT,viewT}
    coeff_prev::Coefficients{matT,viewT}
    
    ##### quadratic surrogate #####
    proj::Coefficients{matT,viewT}
    res::Residuals{R1,R2,R3}
    grad::Coefficients{matT,viewT}
end

function MVDAProblem{T}(Y, X, vertex, label2vertex, vertex2label, coeff, coeff_prev, proj, res, grad) where T <: Real
    # get type parameters
    matT = typeof(Y)
    labelT = keytype(label2vertex)
    viewT = typeof(coeff.dim)
    R1 = typeof(res.main)
    R2 = typeof(res.dist)
    R3 = typeof(res.weighted)
    
    MVDAProblem{T,matT,labelT,viewT,R1,R2,R3}(
        Y, X, vertex, label2vertex, vertex2label,   # data
        coeff, coeff_prev,                          # model
        proj, res, grad,                            # quadratic surrogate
    )
end

function MVDAProblem(labels, X)
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
        Y, X, vertex, label2vertex, vertex2label,   # data
        coeff, coeff_prev,                          # model
        proj, res, grad,                            # quadratic surrogate
    )
end

"""
Returns the floating point type used for model coefficients.
"""
floattype(::MVDAProblem{T}) where T = T

"""
Returns the number of samples, number of features, and number of categories, respectively.
"""
probdims(problem::MVDAProblem) = size(problem.X, 1), size(problem.X, 2), length(problem.vertex)

"""
Returns the prefactors on main residuals and distance resdiduals.
"""
function get_prefactors(problem::MVDAProblem, rho, k)
    n, p, c = probdims(problem)
    a = 1 / sqrt(n)
    b = sqrt(1 / (p - k + 1))
    return a, b
end

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
end
