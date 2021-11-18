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

function __allocate_problem_arrays__(::Nothing, ::Type{T}, n, p, c, intercept::Bool) where T<:Real
    coeff = __allocate_coeff__(T, p+intercept, c)
    coeff_prev = __allocate_coeff__(T, p+intercept, c)
    proj = __allocate_coeff__(T, p+intercept, c)
    res = __allocate_res__(T, n, p+intercept, c)
    grad = __allocate_coeff__(T, p+intercept, c)
    return coeff, coeff_prev, proj, res, grad
end

function __allocate_problem_arrays__(::Kernel, ::Type{T}, n, p, c, intercept::Bool) where T<:Real
    coeff = __allocate_coeff__(T, n+intercept, c)
    coeff_prev = __allocate_coeff__(T, n+intercept, c)
    proj = __allocate_coeff__(T, n+intercept, c)
    res = __allocate_res__(T, n, n+intercept, c)
    grad = __allocate_coeff__(T, n+intercept, c)
    return coeff, coeff_prev, proj, res, grad
end

function vertex_encoding(::Type{T}, labels, class) where T<:Real
    # dimensions
    c = length(class)

    # enumerate vertices
    a = ( 1 + sqrt(c) ) / ( (c-1)^(3/2) )
    b = sqrt( c / (c-1) )
    vertex = Vector{Vector{T}}(undef, c)
    vertex[1] = 1 / sqrt(c-1) * ones(c-1)
    for j in 2:c
        v = -a * ones(c-1)
        v[j-1] += b
        vertex[j] = v
    end

    # encoding: map labels to vertices and vice-versa
    label2vertex = Dict(class_j => vertex[j] for (j, class_j) in enumerate(class))
    vertex2label = Dict(vertex[j] => class_j for (j, class_j) in enumerate(class))

    # create response matrix based on encoding
    Y = create_Y(T, labels, label2vertex)

    return Y, vertex, label2vertex, vertex2label
end

function create_Y(::Type{T}, labels, label2vertex) where T<:Real
    n, c = length(labels), length(label2vertex)
    Y = Matrix{T}(undef, n, c-1)
    @views for (i, label_i) in enumerate(labels)
        Y[i,:] .= label2vertex[label_i]
    end
    return Y
end

function create_X_and_K(kernel::Kernel, ::Type{T}, data, intercept) where T<:Real
    K = kernelmatrix(kernel, data, obsdim=1)
    intercept && (K = [K ones(size(K, 1))])
    X = copy(data)
    return X, K
end

function create_X_and_K(kernel::Nothing, ::Type{T}, data, intercept) where T<:Real
    X = intercept ? [data ones(size(data, 1))] : copy(data)
    return X, nothing
end

struct MVDAProblem{T<:AbstractFloat,kernT,matT<:AbstractMatrix{T},labelT,viewT,R1,R2,R3}
    ##### dimensions #####
    n::Int
    p::Int
    c::Int
    
    ##### data #####
    Y::matT
    X::matT
    K::Union{Nothing,matT}
    vertex::Vector{Vector{T}}
    label2vertex::Dict{labelT,Vector{T}}
    vertex2label::Dict{Vector{T},labelT}
    intercept::Bool

    ##### model #####
    kernel::kernT
    coeff::Coefficients{matT,viewT}
    coeff_prev::Coefficients{matT,viewT}
    
    ##### quadratic surrogate #####
    proj::Coefficients{matT,viewT}
    res::Residuals{R1,R2,R3}
    grad::Coefficients{matT,viewT}

    ##### Inner Constructor #####
    function MVDAProblem{T}(n, p, c, Y, X, K, 
        vertex, label2vertex, vertex2label, intercept,
        kernel, coeff, coeff_prev,
        proj, res, grad) where T <: Real
        # get type parameters
        kernT = typeof(kernel)
        matT = typeof(Y)
        labelT = keytype(label2vertex)
        viewT = typeof(coeff.dim)
        R1 = typeof(res.main)
        R2 = typeof(res.dist)
        R3 = typeof(res.weighted)
        
        new{T,kernT,matT,labelT,viewT,R1,R2,R3}(
            n, p, c,
            Y, X, K, vertex, label2vertex, vertex2label, intercept,
            kernel, coeff, coeff_prev,
            proj, res, grad,
        )
    end
end

function MVDAProblem(labels, data; intercept::Bool=true, kernel::Union{Nothing,Kernel}=nothing)
    # get problem info
    class = sort!(unique(labels))
    (n, p), c = size(data), length(class)
    T = Float64 # TODO

    # create design matrices
    X, K = create_X_and_K(kernel, T, data, intercept)

    # assign classes to vertices and create response matrix
    Y, vertex, label2vertex, vertex2label = vertex_encoding(T, labels, class)

    # allocate data structures for coefficients, projections, residuals, and gradient
    coeff, coeff_prev, proj, res, grad = __allocate_problem_arrays__(kernel, T, n, p, c, intercept)

    return MVDAProblem{T}(
        n, p, c,
        Y, X, K, vertex, label2vertex, vertex2label, intercept,
        kernel, coeff, coeff_prev,
        proj, res, grad,
    )
end

function remake(problem::MVDAProblem, labels, data)
    # extract encoding-dependent fields + problem info
    @unpack vertex, label2vertex, vertex2label, intercept, kernel = problem
    n, p, c = length(labels), problem.p, problem.c
    T = floattype(problem)

    # create new design and response matrices
    X, K = create_X_and_K(kernel, T, data, intercept)
    Y = create_Y(T, labels, label2vertex)

    # allocate data structures for coefficients, projections, residuals, and gradient
    coeff, coeff_prev, proj, res, grad = __allocate_problem_arrays__(kernel, T, n, p, c, intercept)

    return MVDAProblem{T}(
        n, p, c,
        Y, X, K, vertex, label2vertex, vertex2label, intercept,
        kernel, coeff, coeff_prev,
        proj, res, grad,
    )
end

"""
Return the floating-point type used for model coefficients.
"""
floattype(::MVDAProblem{T}) where T = T

"""
Return the design matrix, `X`, in the linear model `Y ∼ X * B`.
"""
get_design_matrix(problem::MVDAProblem) = get_design_matrix(problem.kernel, problem) # dispatch
get_design_matrix(::Nothing, problem::MVDAProblem) = problem.X  # linear case
get_design_matrix(::Kernel, problem::MVDAProblem) = problem.K   # nonlinear case

"""
Returns the number of samples, number of features, and number of categories, respectively.
"""
probdims(problem::MVDAProblem) = (problem.n, problem.p, problem.c)

predict(problem::MVDAProblem, x::AbstractVector) = predict(problem.kernel, problem, x)
predict(problem::MVDAProblem, X::AbstractMatrix) = map(xᵢ -> predict(problem, xᵢ), eachrow(X))
predict(::Nothing, problem::MVDAProblem, x::AbstractVector) = problem.proj.all' * x

function predict(::Kernel, problem::MVDAProblem, x::AbstractVector)
    _, _, c = probdims(problem)
    κ = problem.kernel
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

classify(problem::MVDAProblem, X::AbstractMatrix) = map(xᵢ -> classify(problem, xᵢ), eachrow(X))

function classify(problem::MVDAProblem, x::AbstractVector)
    y = predict(problem, x)
    v = problem.vertex
    distances = [norm(y - v[j]) for j in eachindex(v)]
    j = argmin(distances)
    return problem.vertex2label[v[j]]
end

function Base.show(io::IO, problem::MVDAProblem)
    n, p, c = probdims(problem)
    T = floattype(problem)
    kernT = typeof(problem.kernel)
    respT = eltype(problem.Y)
    matT = eltype(problem.X)
    labelT = keytype(problem.label2vertex)
    kernel_info = kernT <: Nothing ? "linear classifier" : "nonlinear classifier ($(kernT))"
    print(io, "MVDAProblem{$(T)}")
    print(io, "\n  ∘ $(kernel_info)")
    print(io, "\n  ∘ $(n) sample(s) ($(respT))")
    print(io, "\n  ∘ $(p) feature(s) ($(matT))")
    print(io, "\n  ∘ $(c) categories ($(labelT))")
    print(io, "\n  ∘ intercept? $(problem.intercept)")
end
