struct SimplexBased{T<:AbstractVector,K} <: VectorLabelEncoding{T,K}
    vertex::Vector{T}

    function SimplexBased{T,K}() where {T<:AbstractVector{<:Number},K}
        # enumerate vertices
        A = (1 + sqrt(K)) / ((K-1)^(3/2))
        B = sqrt(K/(K-1))
        vertex = Vector{T}(undef, K)
        vertex[1] = fill!(similar(T, K-1), 1 / sqrt(K-1))
        for j in 2:K
            v = fill!(similar(T, K-1), -A)
            v[j-1] += B
            vertex[j] = v
        end
        if K == 2
            vertex[1] = round.(vertex[1])
            vertex[2] = round.(vertex[2])
        end
        return new{T,K}(vertex)
    end
end

SimplexBased(K::Integer) = SimplexBased{Vector{Float64},K}()
SimplexBased(::Type{T}, K::Integer) where T<:Number = SimplexBased{Vector{T},K}()

function MLDataUtils.label2ind(lbl::AbstractVector, lm::SimplexBased{T,K};
    buffer::AbstractVector=similar(lbl)) where {T,K}
    #
    best_j = 0
    for j in 1:K
        @. buffer = lbl - lm.vertex[j]
        if isapprox(norm(buffer), 0)
            best_j = j
            break
        end
    end
    if best_j == 0
        error("Vertex $(lbl) not found in encoding. Did you mean to call `classify(encoding, y)`?")
    end
    return best_j
end

MLDataUtils.ind2label(j::Integer, lm::SimplexBased) = lm.vertex[j]
MLDataUtils.label(lm::SimplexBased) = lm.vertex

MLDataUtils.isposlabel(value, lm::SimplexBased{T,2})  where {T} = label2ind(value, lm) == 1
MLDataUtils.isneglabel(value, lm::SimplexBased{T,2})  where {T} = label2ind(value, lm) == 2

function MLDataUtils.islabelenc(target::AbstractVector{<:Real}, lm::SimplexBased; strict::Bool=true) where K
    if strict
        target in lm.vertex
    else
        length(target) == (K-1) && isapprox(norm(target), 1)
    end
end

MLDataUtils.islabelenc(targets::AbstractVector{<:AbstractVector{<:Real}}, lm::SimplexBased; kwargs...) = all(islabelenc(y, lm; kwargs...) for y in targets)
MLDataUtils.islabelenc(targets::AbstractMatrix{<:Real}, lm::SimplexBased; obsdim=1, kwargs...) = islabelenc(targets, lm, convert(MLDataUtils.LearnBase.ObsDimension, obsdim), kwargs...)
MLDataUtils.islabelenc(targets::AbstractMatrix{<:Real}, lm::SimplexBased, ::ObsDim.Constant{1}, kwargs...) = all(islabelenc(y, lm; kwargs...) for y in eachrow(targets))
MLDataUtils.islabelenc(targets::AbstractMatrix{<:Real}, lm::SimplexBased, ::ObsDim.Constant{2}, kwargs...) = all(islabelenc(y, lm; kwargs...) for y in eachcol(targets))

function MLDataUtils.classify(lm::SimplexBased{T,K}, y::AbstractVector{<:Real};
    buffer::AbstractVector{<:Real}=similar(y)) where {T,K}
    #
    best_j, best_distance = 0, Inf
    for j in 1:K
        @. buffer = y - lm.vertex[j]
        distance = norm(buffer)
        if distance < best_distance
            best_j, best_distance = j, distance
        end
    end
    return copy(lm.vertex[best_j])
end

MLDataUtils.classify!(buffer::AbstractVector{<:Real}, lm::SimplexBased, y::AbstractVector{<:Real}; kwargs...) = copyto!(buffer, classify(lm, y; kwargs...))

function nearest_vertex_index(lm::SimplexBased{T,K}, y::AbstractVector{<:Real};
    buffer::AbstractVector{<:Real}=similar(y)) where {T,K}
    #
    best_j, best_distance = 0, Inf
    for j in 1:K
        @. buffer = y - lm.vertex[j]
        distance = norm(buffer)
        if distance < best_distance
            best_j, best_distance = j, distance
        end
    end
    return best_j
end