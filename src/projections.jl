"""
Project `x` onto sparsity set with `k` non-zero elements.
Assumes `idx` enters as a vector of indices into `x`.
"""
function project_l0_ball!(x::AbstractVector, k::Integer,
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
        sample!(_indexes_, _buffer_, replace=false)
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

struct L0Projection <: Function
    idx::Vector{Int}
    buffer::Vector{Int}

    function L0Projection(n::Int)
        new(collect(1:n), Vector{Int}(undef, n))
    end
end

(P::L0Projection)(x, k) = project_l0_ball!(x, k, P.idx, P.buffer)

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
#   HomogeneousL0Projection
#
#   Scores matrix rows or columns by their norms to induce sparsity.
#
struct HomogeneousL0Projection <: Function
    idx::Vector{Int}
    buffer::Vector{Int}
    scores::Vector{Float64}

    function HomogeneousL0Projection(n::Int)
        new(collect(1:n), Vector{Int}(undef, n), Vector{Float64}(undef, n))
    end
end

(P::HomogeneousL0Projection)(X::AbstractMatrix, k) = project_l0_ball!(X, k, P.scores, P.idx, P.buffer; obsdim=ObsDim.Constant{1}())
