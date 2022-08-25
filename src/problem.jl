const Coefficients{T1,T2} = NamedTuple{(:slope,:intercept), Tuple{T1,T2}}

allocate_coeff(::Type{T}, ::Nothing, n::Integer, p::Integer, c::Integer) where T<:AbstractFloat = (; slope=zeros(T, p, c-1), intercept=zeros(T, c-1))
allocate_coeff(::Type{T}, ::Kernel, n::Integer, p::Integer, c::Integer) where T<:AbstractFloat = (; slope=zeros(T, n, c-1), intercept=zeros(T, c-1))

const Residuals{T1,T2} = NamedTuple{(:loss,:dist), Tuple{T1,T2}}

allocate_res(::Type{T}, ::Nothing, n::Integer, p::Integer, c::Integer) where T<:AbstractFloat = (; loss=zeros(T, n, c-1), dist=zeros(T, p, c-1))
allocate_res(::Type{T}, ::Kernel, n::Integer, p::Integer, c::Integer) where T<:AbstractFloat = (; loss=zeros(T, n, c-1), dist=zeros(T, n, c-1))

function create_X_and_K(kernel::Kernel, ::Type{T}, data) where T<:AbstractFloat
    X = copyto!(similar(data, T), data)
    K = kernelmatrix(kernel, data, obsdim=1)
    return (X, K)
end

function create_X_and_K(::Nothing, ::Type{T}, data) where T<: AbstractFloat
    X = copyto!(similar(data, T), data)
    K = nothing
    return (X, K)
end

"""
    MVDAProblem{T} where T<:AbstractFloat

Representation of a vertex discriminant analysis problem using floating-point type `T`.
Each unique class label is mapped to a unique vertex of a regular simplex. A `(c-1)`-simplex
is used to encode `c` classes.

**Note**: Matrix `X` is used as the design matrix in linear problems (`kernel isa Nothing`);
otherwise the matrix `K` is used for the design (`kernel isa Kernel`).

# Fields

- `n`: Number of samples/instances in problem.
- `p`: Number of features in each sample/instance.
- `c`: Number of classes represented in vertex space.

- `encoding`: A type representing the simplex vertices used to represent labels.
- `kernel`: A `Kernel` object from KernelFunctions.jl. See also: [`Kernel`](@ref).
  (default: `kernel=nothing`)
- `labels`: An ordered list of labels in which labels[i] corresponds to the i-th vertex in
  the `encoding`.
- `intercept`: Indicates whether the design matrix/model include an intercept term.
  (default: `intercept=true`)

- `Y`: Response matrix. Each row `Y[i,:]` corresponds to the vertex assigned to sample `i`.
- `X`: Data matrix. Each row `X[i,:]` corresponds to a sample `i` with `p` features.
- `K`: Kernel matrix. Each row `K[i,:]` corresponds to a sample `i`.

- `coeff`: A `NamedTuple` containing the current estimates for the model. Split into
  `coeff.slope` and `coeff.intercept`.
- `coeff_prev`: Similar to `coeff`, but contains previous estimates.
- `coeff_proj`: Similar to `coeff`, but contains projection of `coeff`.
- `res`: A `NamedTuple` containing residuals for the loss (`coeff.loss`) and distance
  penalty (`coeff.dist`).
- `grad`: Similar to `coeff`, but contains gradient with respect to `coeff`.
"""
struct MVDAProblem{T<:AbstractFloat,encT,kernT,labelT,matT,vecT}
    ##### dimensions #####
    n::Int
    p::Int
    c::Int
    
    ##### settings #####
    encoding::encT
    kernel::kernT
    labels::labelT
    intercept::Bool

    ##### data #####
    Y::matT
    X::matT
    K::Union{Nothing,matT}

    ##### model #####
    coeff::Coefficients{matT,vecT}
    coeff_prev::Coefficients{matT,vecT}
    coeff_proj::Coefficients{matT,vecT}

    ##### loss model #####
    res::Residuals{matT,matT}
    grad::Coefficients{matT,vecT}

    ##### Inner Constructor #####
    function MVDAProblem{T}(data_L, data_X, encoding::encT, kernel::kernT, labels::labelT, intercept::Bool) where {T,encT,kernT,labelT}
        # Get problem dimensions: samples, features, classes
        ((n, p), c) = size(data_X), nlabel(encoding)

        # Sanity checks.
        if length(labels) != c
            error("Number of vertices ($c) does not match number of vertex labels $(length(labels)).")
        end
        if any(!(l in labels) for l in data_L)
            error("Found new labels not in label set: $(setdiff(unique(data_L), labels))")
        end

        # Create the data matrices.
        X, K = create_X_and_K(kernel, T, data_X)
        Y = similar(X, n, c-1)
        for (i, l) in enumerate(data_L)
            j = findfirst(isequal(l), labels)
            Y[i,:] = ind2label(j, encoding)
        end

        # Allocate coefficients, residuals, and gradients.
        coeff = allocate_coeff(T, kernel, n, p, c)
        coeff_prev = allocate_coeff(T, kernel, n, p, c)
        coeff_proj = allocate_coeff(T, kernel, n, p, c)
        res = allocate_res(T, kernel, n, p, c)
        grad = allocate_coeff(T, kernel, n, p, c)

        # Extract the missing type information.
        matT, vecT = typeof(coeff.slope), typeof(coeff.intercept)

        new{T,encT,kernT,labelT,matT,vecT}(
            n, p, c,
            encoding, kernel, labels, intercept,
            Y, X, K,
            coeff, coeff_prev, coeff_proj,
            res, grad
        )
    end
end

"""
    MVDAProblem{T}(labels, data; [intercept=true], [kernel=nothing])

Create a `MVDAProblem` instance from the labeled dataset `(label, data)`.

Floating point numbers are set to type `T`.
The `labels` should enter as an iterable object, and `data` should be a `n × p` matrix with
samples/instances aligned along rows (e.g. `data[i,:]` is sample `i`).

!!! note

    Defaults to linear classifier, `kernel=nothing`.
    Specifying a `Kernel` requires `using KernelFunctions` first.

# Keyword Arguments

- `intercept`: Should the model include an intercept term?
- `kernel`: How should the data be transformed?.
"""
function MVDAProblem{T}(data_L, data_X;
    intercept::Bool=true,
    kernel::Union{Nothing,Kernel}=nothing) where T
    #
    labels = sort!(unique(data_L))
    K = length(labels)
    encoding = SimplexBased{Vector{T},K}()
    MVDAProblem{T}(data_L, data_X, encoding, kernel, labels, intercept)
end

"""
    MVDAProblem(labels, data; kwargs...)

Create a `MVDAProblem` that defaults to `Float64` floating point numbers.
"""
MVDAProblem(data_L, data_X; kwargs...) = MVDAProblem{Float64}(data_L, data_X; kwargs...)

"""
    MVDAProblem(labels, data, old::MVDAProblem)

Create a new `MVDAProblem` using the settings of the `old` one.

Specifically, the new problem with use the same encoding and label set to keep `classify`
results consistent with the input model.
"""
MVDAProblem(data_L, data_X, old::MVDAProblem) = MVDAProblem{floattype(old)}(data_L, data_X, old.encoding, old.kernel, old.labels, old.intercept)

"""
    MVDAProblem(labels, data, old::MVDAProblem, kernel::Kernel)

Create a new `MVDAProblem` using a different kernel but with the settings of the `old` one.

Specifically, the new problem with use the same encoding and label set to keep `classify`
results consistent with the input model.
"""
MVDAProblem(data_L, data_X, old::MVDAProblem, kernel::Kernel) = MVDAProblem{floattype(old)}(data_L, data_X, old.encoding, kernel, old.labels, old.intercept)

MVDAProblem(data_L, data_X, old::MVDAProblem, ::Nothing) = MVDAProblem{floattype(old)}(data_L, data_X, old.encoding, nothing, old.labels, old.intercept)

"""
Return the floating-point type used for model coefficients.
"""
floattype(::MVDAProblem{T}) where T = T

"""
Map vertex labels back to the original label space.
"""
function original_labels(problem::MVDAProblem)
    @unpack Y, encoding, labels = problem
    idx = map(Base.Fix2(label2ind, encoding), eachrow(Y))
    L = map(Base.Fix1(getindex, labels), idx)
    return L
end

"""
Return the design matrix used for fitting a classifier.

Uses `problem.X` in the linear case and `problem.K` in the nonlinear case.
"""
design_matrix(problem::MVDAProblem) = design_matrix(problem.kernel, problem) # dispatch
design_matrix(::Nothing, problem::MVDAProblem) = problem.X                   # linear case
design_matrix(::Kernel, problem::MVDAProblem) = problem.K                    # nonlinear case

"""
Returns the number of samples, number of features, and number of categories, respectively.
"""
probdims(problem::MVDAProblem) = (problem.n, problem.p, problem.c)

function Base.show(io::IO, problem::MVDAProblem)
    n, p, c = probdims(problem)
    T = floattype(problem)
    kernT = typeof(problem.kernel)
    respT = eltype(problem.Y)
    matT = eltype(problem.X)
    labelT = eltype(problem.labels)
    kernel_info = kernT <: Nothing ? "linear classifier" : "nonlinear classifier ($(kernT))"
    print(io, "MVDAProblem{$(T)}")
    print(io, "\n  ∘ $(kernel_info)")
    print(io, "\n  ∘ $(n) sample(s) ($(respT))")
    print(io, "\n  ∘ $(p) feature(s) ($(matT))")
    print(io, "\n  ∘ $(c) categories ($(labelT))")
    print(io, "\n  ∘ intercept? $(problem.intercept)")
end

function maximum_deadzone(problem::MVDAProblem)
    c = problem.c
    return 1//2 * sqrt(2*c/(c-1))
end

"""
    predict(problem::MVDAProblem, x)

When `x` is a vector, predict a value in vertex space for a sample/instance `x` based on the
fitted model in `problem`. Otherwise if `x` is a matrix then each sample is assumed to be
aligned along rows (e.g. `x[i,:]` is sample `i`).

See also: [`classify`](@ref)
"""
MLDataUtils.predict(problem::MVDAProblem, x) = predict(problem.kernel, problem, x)

function MLDataUtils.predict(::Nothing, problem::MVDAProblem, x::AbstractVector)
    @unpack c, coeff_proj, intercept = problem
    B, b0 = coeff_proj.slope, coeff_proj.intercept
    T = floattype(problem)
    y = similar(x, T, c-1)
    if intercept
        copyto!(y, b0)
    else
        fill!(y, 0)
    end
    BLAS.gemv!('T', one(T), B, x, one(T), y)
    return y
end

function MLDataUtils.predict(::Nothing, problem::MVDAProblem, X::AbstractMatrix)
    @unpack c, coeff_proj, intercept = problem
    B, b0 = coeff_proj.slope, coeff_proj.intercept
    T = floattype(problem)
    n = size(X, 1)
    Y = similar(X, T, n, c-1)
    if intercept
        foreach(Base.Fix2(copyto!, b0), eachrow(Y))
    else
        fill!(Y, 0)
    end
    BLAS.gemm!('N', 'N', one(T), X, B, one(T), Y)
    return Y
end

function MLDataUtils.predict(kernel::Kernel, problem::MVDAProblem, x::AbstractVector)
    @unpack c, coeff_proj, intercept = problem
    B, b0 = coeff_proj.slope, coeff_proj.intercept
    T = floattype(problem)
    y = similar(x, T, c-1)
    K = kernelmatrix(kernel, problem.X, x', obsdim=1)
    if intercept
        copyto!(y, b0)
    else
        fill!(y, 0)
    end
    BLAS.gemm!('T', 'N', one(T), B, K, one(T), y)
    return y
end

function MLDataUtils.predict(kernel::Kernel, problem::MVDAProblem, X::AbstractMatrix)
    @unpack c, coeff_proj, intercept = problem
    B, b0 = coeff_proj.slope, coeff_proj.intercept
    T = floattype(problem)
    n = size(X, 1)
    Y = similar(X, T, n, c-1)
    K = kernelmatrix(kernel, X, problem.X, obsdim=1)
    if intercept
        foreach(Base.Fix2(copyto!, b0), eachrow(Y))
    else
        fill!(Y, 0)
    end
    BLAS.gemm!('N', 'N', one(T), K, B, one(T), Y)
    return Y
end

"""
    classify(problem::MVDAProblem, x)

Classify the samples/instances in `x` based on the model in `problem`.

If `x` is a vector then it is treated as an instance. Otherwise if `x` is a matrix then each
sample is assumed to be aligned along rows (e.g. `x[i,:]` is sample `i`).

See also: [`predict`](@ref)
"""
MLDataUtils.classify(problem::MVDAProblem, x) = __classify__(problem, predict(problem, x))

function __classify__(problem::MVDAProblem, y::AbstractVector; kwargs...)
    @unpack encoding, labels = problem
    j = nearest_vertex_index(encoding, y; kwargs...)
    return labels[j]
end

function __classify__(problem::MVDAProblem, Y::AbstractMatrix)
    @unpack encoding, labels = problem
    n = size(Y, 1)
    L = Vector{eltype(labels)}(undef, n)
    num_julia_threads = Threads.nthreads()

    if num_julia_threads == 1
        buffer = similar(y)
        for i in axes(Y, 1)
            y = view(Y, i, :)
            L[i] = __classify__(problem, y, buffer=buffer)
        end
    else
        workers = [similar(Y, size(Y, 2)) for _ in 1:num_julia_threads]
        num_BLAS_threads = BLAS.get_num_threads()
        try
            BLAS.set_num_threads(1)
            @batch per=core for i in axes(Y, 1)
                y = view(Y, i, :)
                local_buffer = workers[Threads.threadid()]
                L[i] = __classify__(problem, y, buffer=local_buffer)
            end
        finally
            BLAS.set_num_threads(num_BLAS_threads)
        end
    end

    return L
end

function confusion_matrix(problem::MVDAProblem, data::Tuple{LT,XT}) where {LT,XT}
    @unpack c, labels = problem
    L, X = data
    Lhat = classify(problem, X)
    d = Dict(label => i for (i, label) in enumerate(labels))
    C = zeros(c, c)
    for (lhat, l) in zip(Lhat, L)
        # rows: true, cols: predicted
        i, j = d[l], d[lhat]
        C[i,j] += 1
    end
    return (C, d)
end

function prediction_probabilities(problem::MVDAProblem, data::Tuple{LT,XT}) where {LT,XT}
    C, d = confusion_matrix(problem, data)
    S = sum(C, dims=1)
    return (C ./ S, d)
end

function confusion_matrix(problem::MVDAProblem, coeff0::AbstractMatrix)
    coeff = problem.coeff_proj.slope
    C = zeros(2, 2)
    for i in axes(coeff0, 1)
        xi = norm(view(coeff, i, :))
        yi = norm(view(coeff0, i, :))
        C[1,1] += (xi != 0) && (yi != 0) # TP
        C[2,1] += (xi != 0) && (yi == 0) # FP
        C[2,1] += (xi == 0) && (yi != 0) # FN
        C[2,2] += (xi == 0) && (yi == 0) # TN
    end
    return C
end

confusion_matrix(problem::MVDAProblem, coeff0::Coefficients) = confusion_matrix(problem, coeff0.slope)

function accuracy(problem::MVDAProblem, data::Tuple{LT,XT}) where {LT,XT}
    L, X = data

    # Sanity check: Y and X have the same number of rows.
    length(L) != size(X, 1) && error("Labels ($(length(L))) not compatible with data X ($(size(X))).")

    Lhat = classify(problem, X)

    # Sweep through predictions and tally the mistakes.
    ncorrect = 0
    for i in eachindex(L)
        ncorrect += (Lhat[i] == L[i])
    end

    return ncorrect / length(L)
end

function save_model(dir::AbstractString, problem::MVDAProblem)
    @unpack coeff, coeff_proj = problem
    writedlm(joinpath(dir, "coef.slope"), coeff.slope, '\t')
    writedlm(joinpath(dir, "coef.intercept"), coeff.intercept, '\t')
    writedlm(joinpath(dir, "proj.slope"), coeff.slope, '\t')
    writedlm(joinpath(dir, "proj.intercept"), coeff.intercept, '\t')
    return nothing
end

function load_model(dir::AbstractString)
    return (;
        coeff=(;
            slope=readdlm(joinpath(dir, "coef.slope")),
            intercept=readdlm(joinpath(dir, "coef.intercept")),
        ),
        coeff_proj=(;
            slope=readdlm(joinpath(dir, "proj.slope")),
            intercept=readdlm(joinpath(dir, "proj.intercept")),
        ),
    )
end

function count_active_variables(problem::MVDAProblem)
    has_nonzero_norm(x) = !isequal(norm(x), 0)
    coeff = problem.coeff_proj.slope
    count(has_nonzero_norm, eachrow(coeff))
end

function active_variables(problem::MVDAProblem)
    coeff = problem.coeff_proj.slope
    idx = Int[]
    for (i, x) in enumerate(eachrow(coeff))
        if norm(x) != 0
            push!(idx, i)
        end
    end
    return idx
end
