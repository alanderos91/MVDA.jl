#
#   L0-type projections
#

"""
Project `x` onto sparsity set with `k` non-zero elements.
Assumes `idx` enters as a vector of indices into `x`.
"""
function project_l0_ball!(x::AbstractVector, k::Integer,
    rng::AbstractRNG=Random.GLOBAL_RNG,
    idx::T=collect(eachindex(x)),
    buffer::T=similar(idx)) where T <: AbstractVector{<:Integer}
    #
    is_equal_magnitude(x, y) = abs(x) == abs(y)
    is_equal_magnitude(y) = Base.Fix2(is_equal_magnitude, y)
    #
    n = length(x)
    # do nothing if k > length(x)
    if k ≥ n return x end
    
    # fill with zeros if k ≤ 0
    if k ≤ 0 return fill!(x, 0) end
    
    pivot = __search_by_partialsort__!(idx, x, k, true)

    # preserve the top k elements
    nonzero_count = __threshold__!(x, abs(pivot))

    # resolve ties
    if nonzero_count > k
        number_to_drop = nonzero_count - k
        _indexes_ = findall(is_equal_magnitude(pivot), x)
        _buffer_ = view(buffer, 1:number_to_drop)
        sample!(rng, _indexes_, _buffer_, replace=false)
        @inbounds for i in _buffer_
            x[i] = 0
        end
    end

    return x
end

"""
Search `x` for the pivot that splits the vector into the `k`-largest elements in magnitude.

The search preserves signs and returns `x[k]` after partially sorting `x`.
"""
function __search_by_partialsort__!(idx, x, k, rev::Bool)
    #
    # Based on https://github.com/JuliaLang/julia/blob/788b2c77c10c2160f4794a4d4b6b81a95a90940c/base/sort.jl#L863
    # This eliminates a mysterious allocation of ~48 bytes per call for
    #   sortperm!(idx, x, alg=algorithm, lt=isless, by=abs, rev=true, initialized=false)
    # where algorithm = PartialQuickSort(lo:hi)
    # Savings are small in terms of performance but add up for CV code.
    #
    lo = k
    hi = k+1

    # Order arguments
    lt  = isless
    by  = abs
    # rev = true
    o = Base.Order.Forward
    order = Base.Order.Perm(Base.Sort.ord(lt, by, rev, o), x)

    # sort!(idx, lo, hi, PartialQuickSort(k), order)
    Base.Sort.Float.fpsort!(idx, PartialQuickSort(lo:hi), order)

    return x[idx[k]]
end

function __threshold__!(x, abs_pivot)
    nonzero_count = 0
    for i in eachindex(x)
        if x[i] == 0 continue end
        if abs(x[i]) < abs_pivot
            x[i] = 0
        else
            nonzero_count += 1
        end
    end
    return nonzero_count
end

__iterateby__(::ObsDim.Constant{1}) = eachrow
__iterateby__(::ObsDim.Constant{2}) = eachcol

function project_l0_ball!(X::AbstractMatrix, k, args...;
    obsdim::T=ObsDim.Constant{1}(),
    rankby::F=Base.Fix2(norm, 2)) where {T,F}
    #
    iterateby = __iterateby__(convert(ObsDimension, obsdim))
    project_l0_ball!(iterateby, rankby, X, k, args...)
end

function project_l0_ball!(itr::F1, f::F2, X::AbstractMatrix, k, scores::S, args...) where {F1,F2,S}
    for (i, xi) in enumerate(itr(X))
        scores[i] = f(xi)
    end
    project_l0_ball!(scores, k, args...)
    for (score_i, x_i) in zip(scores, itr(X))
        if score_i == 0
            fill!(x_i, 0)
        end
    end
    return X
end

#
#   L0Projection
#
#   Induce sparsity on components of a vector. Treats matrices as vectors.
#
struct L0Projection{RNG}
    rng::RNG
    idx::Vector{Int}
    buffer::Vector{Int}

    function L0Projection(rng::RNG, n::Int) where RNG <: AbstractRNG
        new{RNG}(rng, collect(1:n), Vector{Int}(undef, n))
    end
end

(P::L0Projection)(x::AbstractVector, k) = project_l0_ball!(x, k, P.rng, P.idx, P.buffer)
(P::L0Projection)(x::AbstractMatrix, k) = project_l0_ball!(vec(x), k, P.rng, P.idx, P.buffer)

#
#   HomogeneousL0Projection
#
#   Scores matrix rows or columns by their norms to induce sparsity.
#   Assumes that categories are determined from the same set of features.
#
struct HomogeneousL0Projection{RNG}
    rng::RNG
    idx::Vector{Int}
    buffer::Vector{Int}
    scores::Vector{Float64}

    function HomogeneousL0Projection(rng::RNG, n::Int) where RNG <: AbstractRNG
        new{RNG}(rng, collect(1:n), Vector{Int}(undef, n), Vector{Float64}(undef, n))
    end
end

(P::HomogeneousL0Projection)(X::AbstractMatrix, k) = project_l0_ball!(X, k, P.scores, P.rng, P.idx, P.buffer; obsdim=ObsDim.Constant{1}())

#
#   HeterogeneousL0Projection
#
#   Carries out L0Projection on each row or each column of a matrix.
#   Assumes that categories are determined by distinct sets of features.
#
struct HeterogeneousL0Projection{RNG}
    rng::RNG
    projection::Vector{L0Projection{RNG}}

    function HeterogeneousL0Projection(rng::RNG, ncategories::Int, nfeatures::Int) where RNG <: AbstractRNG
        # keep reference to underlying RNG; note that it is shared across each projection operator!
        new{RNG}(rng, [L0Projection(rng, nfeatures) for _ in 1:ncategories])
    end
end

function (P::HeterogeneousL0Projection)(X::AbstractMatrix, k)
    for (j, x) in enumerate(eachcol(X))
        P_j = P.projection[j]
        P_j(x, k)
    end
    return X
end

#
#   make_projection()
#
make_projection(::Type{Nothing}, rng, p, c) = nothing
make_projection(::Type{L0Projection}, rng, p, c) = L0Projection(rng, p*c)
make_projection(::Type{HomogeneousL0Projection}, rng, p, c) = HomogeneousL0Projection(rng, p)
make_projection(::Type{HeterogeneousL0Projection}, rng, p, c) = HeterogeneousL0Projection(rng, c, p)

#
#   apply_projection()
#
"""
Apply a projection to model coefficients.
"""
function apply_projection(problem, extras, hparams, inplace)
    @unpack coeff, coeff_proj = problem
    if inplace
        apply_projection(extras.projection, coeff.slope, hparams)
    else
        copyto!(coeff_proj.slope, coeff.slope)
        copyto!(coeff_proj.intercept, coeff.intercept)
        apply_projection(extras.projection, coeff_proj.slope, hparams)
    end
end

apply_projection(::Nothing, x, hparams) = x

function apply_projection(P::L0Projection, x, hparams)
    @unpack k = hparams
    P(x, k)
end

function apply_projection(P::HomogeneousL0Projection, x, hparams)
    @unpack k = hparams
    P(x, k)
end

function apply_projection(P::HeterogeneousL0Projection, x, hparams)
    @unpack k = hparams
    P(x, k)
end

#
# get_scale_factor()
#
function get_scale_factor(::L0Projection, x, hparams)
    @unpack k = hparams
    scale_factor = 1 / max(1, (prod(size(x)) - k))
    return scale_factor
end

function get_scale_factor(::HomogeneousL0Projection, x, hparams)
    @unpack k = hparams
    m, n = size(x)
    scale_factor = 1 / max(1, n*(m-k))
    return scale_factor
end

function get_scale_factor(::HeterogeneousL0Projection, x, hparams)
    @unpack k = hparams
    m, n = size(x)
    scale_factor = 1 / max(1, n*(m-k))
    return scale_factor
end
