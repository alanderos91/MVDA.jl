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
end

function (P::ApplyL0Projection)(x::AbstractVector, k)
    project_l0_ball!(x, P.idx, k)
end

function (P::ApplyL0Projection)(X::AbstractMatrix, k; on::Symbol=:col, intercept::Bool=true)
    n, m = size(X)
    if on == :col
        # apply projection on columns
        for j in 1:m
            xⱼ = view(X, 1:n-intercept, j)
            P(xⱼ, k[j])
        end
    elseif on == :row
        # apply projection on rows
        for i in 1:n-intercept
            xᵢ = view(X, i, :)
            P(xᵢ, k[i])
        end
    end
    return X
end
