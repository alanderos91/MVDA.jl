"""
Project `x` onto sparsity set with `k` non-zero elements.
Assumes `idx` enters as a vector of indices into `x`.
"""
function project_l0_ball!(x, idx, k)
    #
    #   do nothing if k > length(x)
    #
    if k ≥ length(x) return x end
    
    #
    #   fill with zeros if k ≤ 0
    #
    if k ≤ 0 return fill!(x, 0) end
    
    #
    # find the spliting element
    #
    pivot = l0_search_partialsort!(idx, x, k)
    
    #
    # apply the projection
    #
    kcount = 0
    @inbounds for i in eachindex(x)
        if abs(x[i]) <= abs(pivot) || kcount ≥ k
            x[i] = 0
        else
            kcount += 1
        end
    end
    
    return x
end

function project_l0_ball!(X::AbstractMatrix, idx, scores, k; by::Union{Val{:row}, Val{:col}}=Val(:row))
    if by isa Val{:row}
        n = size(X, 1)
        itr = axes(X, 1)
        itr2 = eachrow(X)
        f = i -> norm(view(X, i, :))
    elseif by isa Val{:col}
        n = size(X, 2)
        itr = axes(X, 2)
        itr2 = eachcol(X)
        f = i -> norm(view(X, :, i))
    else
        error("uncrecognized option `by=$(by)`.")
    end

    # do nothing if k > length(x)
    if k ≥ n return X end
    
    # fill with zeros if k ≤ 0
    if k ≤ 0 return fill!(X, 0) end

    # map rows to a score used in ranking; here we use Euclidean norms
    map!(f, scores, itr)

    # partially sort scores to find the top k rows
    pivot = l0_search_partialsort!(idx, scores, k)

    # apply the projection
    kcount = 0
    @inbounds for (scoreᵢ, xᵢ) in zip(scores, itr2)
        # row is not in the top k
        if scoreᵢ ≤ pivot || kcount ≥ k
            fill!(xᵢ, 0)
        else # row is in the top k
            kcount += 1
        end
    end

    return X
end

"""
Search `x` for the pivot that splits the vector into the `k`-largest elements in magnitude.

The search preserves signs and returns `x[k]` after partially sorting `x`.
"""
function l0_search_partialsort!(idx, x, k)
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
    rev = true
    o = Base.Order.Forward
    order = Base.Order.Perm(Base.Sort.ord(lt, by, rev, o), x)

    # Initialize the idx array; algorithm relies on idx[i] = i
    @inbounds for i in eachindex(idx)
        idx[i] = i
    end

    # sort!(idx, lo, hi, PartialQuickSort(k), order)
    Base.Sort.Float.fpsort!(idx, PartialQuickSort(lo:hi), order)
    
    return x[idx[k+1]]
end

struct ApplyL0Projection <: Function
    idx::Vector{Int}

    function ApplyL0Projection(n::Int)
        new(collect(1:n))
    end
end

function (P::ApplyL0Projection)(x::AbstractVector, k)
    project_l0_ball!(x, P.idx, k)
end

struct ApplyStructuredL0Projection <: Function
    idx::Vector{Int}
    scores::Vector{Float64}

    function ApplyStructuredL0Projection(n::Int)
        idx = collect(1:n)
        scores = zeros(n)
        new(idx, scores)
    end
end

function (P::ApplyStructuredL0Projection)(X::AbstractMatrix, k)
    project_l0_ball!(X, P.idx, P.scores, k, by=Val(:row))
end
