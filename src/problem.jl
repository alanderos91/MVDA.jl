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

"""
    MVDAProblem{T} where T<:AbstractFloat

Representation of a vertex discriminant analysis problem using floating-point type `T`.
Each unique class label is mapped to a unique vertex of a regular simplex. A `(c-1)`-simplex is used to encode `c` classes.

**Note**: The matrix `X` is used as the *design matrix* for linear problems (i.e. `kernel isa Nothing`); otherwise the matrix `K` is used for the design (i.e. `kernel isa Kernel`).
In either case, the *design matrix* will contain a column of `1`s when `intecept=true`.

# Fields

- `n`: Number of samples/instances in problem.
- `p`: Number of features in each sample/instance.
- `c`: Total number of classes represented in vertex space.

- `Y`: Response matrix. Each row `Y[i,:]` corresponds to the vertex assigned to sample `i`.
- `X`: Data matrix. Each row `X[i,:]` corresponds to a sample `i` with `p` features.
- `K`: Kernel matrix. Each row `K[i,:]` corresponds to a sample `i`.

- `vertex`: List of vertices used to encode classes.
- `label2vertex`: An associative map translating a value from original label space to vertex space.
- `vertex2label`: An associative map trasnlating a value from vertex space to the original label space.
- `intercept`: Indicates whether the design matrix/model include an intercept term (`intercept=true`).

- `kernel`: A `Kernel` object from KernelFunctions.jl. See also: [`Kernel`](@ref).
- `coeff`: A `NamedTuple` containing the current estimates for the full model (`coeff.all`) and views along each dimension in vertex space (`coeff.dim`).
- `coeff_prev`: Similar to `coeff`, but contains previous estimates.
- `proj`: Similar to `coeff`, but contains projection of `coeff`.
- `res`: A `NamedTuple` containing various residuals.
- `grad`: Similar to `coeff`, but contains gradient with respect to `coeff`.
"""
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

"""
    MVDAProblem(labels, data; [intercept=true], [kernel=nothing])

Create a `MVDAProblem` instance from the labeled dataset `(label, data)`.

The `label` information should enter as an iterable object, and `data` should be a `n × p` matrix with samples/instances aligned along rows (e.g. `data[i,:]` is sample `i`).

!!! note

    Defaults to linear classifier, `kernel=nothing`.
    Specifying a `Kernel` requires `using KernelFunctions` first.

# Keyword Arguments

- `intercept`: Should the model include an intercept term?
- `kernel`: How should the data be transformed?.
"""
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

"""
    change_data(problem::MVDAProblem, labels::AbstractVector, data)

Create a new `MVDAProblem` instance from the labeled dataset `(label, data)` using the vertex encoding from the reference `problem`.
"""
function change_data(problem::MVDAProblem, labels::AbstractVector, data)
    @unpack label2vertex = problem          # extract encoding-dependent fields + problem info
    T = floattype(problem)
    Y = create_Y(T, labels, label2vertex)   # create response matrix
    change_data(problem, Y, data)           # dispatch
end

"""
    change_data(problem::MVDAProblem, Y::AbstractMatrix, data)

Create a new `MVDAProblem` instance from the labeled dataset `(Y, data)` using the vertex encoding from the reference `problem`.

Assumes `Y` already contains vertex assignments consistent with the encoding in `problem`.
"""
function change_data(problem::MVDAProblem, Y::AbstractMatrix, data)
    # extract encoding-dependent fields + problem info
    @unpack vertex, label2vertex, vertex2label, intercept, kernel = problem
    has_intercept = all(isequal(1), data[:,end])
    n, p, c = size(Y, 1), has_intercept ? size(data, 2)-1 : size(data, 2), problem.c
    T = floattype(problem)

    # create new design matrices
    X, K = create_X_and_K(kernel, T, data, intercept)

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
Return the design matrix used for fitting a classifier.

Uses `problem.X` when `problem.kernel isa Nothing` and `problem.K` when `problem.kernel isa Kernel` from KernelFunctions.jl.
"""
get_design_matrix(problem::MVDAProblem) = __get_design_matrix__(problem.kernel, problem) # dispatch
__get_design_matrix__(::Nothing, problem::MVDAProblem) = problem.X  # linear case
__get_design_matrix__(::Kernel, problem::MVDAProblem) = problem.K   # nonlinear case

"""
Returns the number of samples, number of features, and number of categories, respectively.
"""
probdims(problem::MVDAProblem) = (problem.n, problem.p, problem.c)

"""
    predict(problem::MVDAProblem, x)

When `x` is a vector, predict the vertex value of a sample/instance `x` based on the fitted model in `problem`.
Otherwise if `x` is a matrix then each sample is assumed to be aligned along rows (e.g. `x[i,:]` is sample `i`).

See also: [`classify`](@ref)
"""
predict(problem::MVDAProblem, x) = __predict__(problem.kernel, problem, x)

function __predict__(::Nothing, problem::MVDAProblem, x::AbstractVector)
    @unpack p, proj, intercept = problem
    B = view(proj.all, 1:p, :)
    B0 = view(proj.all, p+intercept, :)
    y = B' * x
    intercept && (y += B0)
    return y
end

function __predict__(::Nothing, problem::MVDAProblem, X::AbstractMatrix)
    @unpack p, proj, intercept = problem
    B = view(proj.all, 1:p, :)
    B0 = view(proj.all, p+intercept, :)
    Y = X * B
    intercept && (Y .+= B0')
    return Y
end

function __predict__(::Kernel, problem::MVDAProblem, x::AbstractVector)
    c = problem.c
    κ = problem.kernel
    Γ = problem.proj.all
    X = problem.X

    ϕ = zeros(floattype(problem), c-1)
    for (i, xᵢ) in enumerate(eachrow(X))
        k = κ(xᵢ, x)
        γ = Γ[i,:]
        @inbounds for j in eachindex(ϕ)
            ϕ[j] = muladd(k, γ[j], ϕ[j])
        end
    end

    # ...and accumulate intercept term, ϕⱼ += Γ₀ⱼ.
    if problem.intercept
        @inbounds for j in eachindex(ϕ)
            ϕ[j] += Γ[end,j]
        end
    end

    return ϕ
end

function __predict__(::Kernel, problem::MVDAProblem, X::AbstractMatrix)
    @unpack c = problem
    n = size(X, 1)
    Y = Matrix{floattype(problem)}(undef, n, c-1)
    nthreads = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    @batch per=core for i in 1:n
        Y[i,:] .= predict(problem, X[i,:])
    end
    BLAS.set_num_threads(nthreads)
    return Y
end

"""
    classify(problem::MVDAProblem, x)

Classify the samples/instances in `x` based on the model in `problem`.

If `x` is a vector then it is treated as an instance.
Otherwise if `x` is a matrix then each sample is assumed to be aligned along rows (e.g. `x[i,:]` is sample `i`).
See also: [`predict`](@ref)
"""
classify(problem::MVDAProblem, x) = __classify__(problem, predict(problem, x))

function __classify__(problem::MVDAProblem, y::AbstractVector)
    @unpack vertex, vertex2label = problem
    distances = [norm(y - vertex[j]) for j in eachindex(vertex)]
    j = argmin(distances)
    return problem.vertex2label[vertex[j]]
end

function __classify__(problem::MVDAProblem, Y::AbstractMatrix)
    n = size(Y, 1)
    L = Vector{valtype(problem.vertex2label)}(undef, n)
    nthreads = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    @batch per=core for i in eachindex(L)
        L[i] = __classify__(problem, Y[i,:])
    end
    BLAS.set_num_threads(nthreads)
    return L
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

function maximal_deadzone(problem::MVDAProblem)
    c = problem.c
    ifelse(c == 2, 0.5, 1//2 * sqrt(2*c/(c-1)))
end

function set_initial_coefficients!(::Nothing, train_coeff, coeff, idx)
    copyto!(train_coeff, coeff)
end

function set_initial_coefficients!(::Kernel, train_coeff, coeff, idx)
    @views for (i, idx_i) in enumerate(idx)
        train_coeff[i, :] .= coeff[idx_i, :]
    end
end

function set_initial_coefficients!(train_problem::MVDAProblem, problem::MVDAProblem, idx)
    set_initial_coefficients!(train_problem.kernel, train_problem.coeff.all, problem.coeff.all, idx)
    set_initial_coefficients!(train_problem.kernel, train_problem.coeff_prev.all, problem.coeff_prev.all, idx)
end
