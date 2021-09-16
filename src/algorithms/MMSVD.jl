"""
Solve least squares problem via SVD.
"""
struct MMSVD <: AbstractMMAlg end

# Initialize data structures.
function __mm_init__(::MMSVD, problem, ::Nothing)
    @unpack X, coeff = problem
    n, p, c = probdims(problem)
    T = floattype(problem)

    # thin SVD of X
    U, s, V = __svd_wrapper__(X)
    r = length(s) # number of nonzero singular values

    # worker arrays
    Z = Matrix{T}(undef, n, c-1)
    buffer = Vector{T}(undef, r)
    
    # diagonal matrices
    Ψ = [Diagonal(Vector{T}(undef, r)) for _ in 1:c-1]
    
    return (;
        apply_projection=ApplyStructuredL0Projection(p),
        U=U, s=s, V=V,
        Z=Z, Ψ=Ψ,
        buffer=buffer,
    )
end

# Check for data structure allocations; otherwise initialize.
function __mm_init__(::MMSVD, problem, extras)
    if :apply_projection in keys(extras) && :buffer in keys(extras) # TODO
        return extras
    else
        __mm_init__(MMSVD(), problem, nothing)
    end
end

# Update data structures due to change in model subsets, k.
__mm_update_sparsity__(::MMSVD, problem, ϵ, ρ, k, extras) = nothing

# Update data structures due to changing ρ.
function __mm_update_rho__(::MMSVD, problem, ϵ, ρ, k, extras)
    @unpack s, Ψ = extras
    n, p, c = probdims(problem)
    a² = 1 / n

    # Update the diagonal matrices Ψⱼ = (a² Σ²) / (a² Σ² + b² I).
    @inbounds for j in eachindex(Ψ)
        Ψⱼ = Ψ[j]
        b² = ρ
        @inbounds for i in eachindex(Ψⱼ.diag)
            sᵢ² = s[i]^2
            Ψⱼ.diag[i] = a² * sᵢ² / (a² * sᵢ² + b²)
        end
    end

    return nothing
end

# Update data structures due to changing λ.
function __mm_update_lambda__(::MMSVD, problem, ϵ, λ, extras)
    @unpack s, Ψ = extras
    n, p, c = probdims(problem)
    a² = 1 / n

    # Update the diagonal matrices Ψⱼ = (a² Σ²) / (a² Σ² + b² I).
    @inbounds for j in eachindex(Ψ)
        Ψⱼ = Ψ[j]
        b² = λ
        @inbounds for i in eachindex(Ψⱼ.diag)
            sᵢ² = s[i]^2
            Ψⱼ.diag[i] = a² * sᵢ² / (a² * sᵢ² + b²)
        end
    end

    return nothing
end

# Apply one update.
function __mm_iterate__(::MMSVD, problem, ϵ, ρ, k, extras)
    @unpack intercept, coeff, proj = problem
    @unpack buffer, apply_projection = extras
    @unpack Z, Ψ, U, s, V = extras
    n, p, c = probdims(problem)
    Σ = Diagonal(s)
    T = floattype(problem)

    # need to compute Z via residuals...
    copyto!(proj.all, coeff.all)
    apply_projection(view(proj.all, 1:p, :), k)
    __evaluate_residuals__(problem, ϵ, extras, true, false, true)

    for j in eachindex(coeff.dim)
        # Rename relevant arrays/views
        βⱼ = coeff.dim[j]
        pⱼ = proj.dim[j]
        zⱼ = view(Z, :, j)
        Ψⱼ = Ψ[j]

        # Update parameters along dimension j:
        # βⱼ = pⱼ + V' * Ψⱼ * (Σ⁻¹Uᵀzⱼ - Vᵀpⱼ) 
        mul!(buffer, U', zⱼ)
        ldiv!(Σ, buffer)
        mul!(buffer, V', pⱼ, -one(T), one(T))
        lmul!(Ψⱼ, buffer)
        mul!(βⱼ, V, buffer)
        axpy!(one(T), pⱼ, βⱼ)
        # @assert βⱼ ≈ pⱼ + V * Ψⱼ * (inv(Σ)*U'*zⱼ - V'*pⱼ)
    end

    return nothing
end

# Apply one update in regularized version.
function __mm_iterate__(::MMSVD, problem, ϵ, λ, extras)
    @unpack intercept, coeff, proj = problem
    @unpack buffer = extras
    @unpack Z, Ψ, U, s, V = extras
    Σ = Diagonal(s)
    T = floattype(problem)

    # need to compute Z via residuals...
    __evaluate_residuals__(problem, ϵ, extras, true, false, true)

    for j in eachindex(coeff.dim)
        # Rename relevant arrays/views
        βⱼ = coeff.dim[j]
        zⱼ = view(Z, :, j)
        Ψⱼ = Ψ[j]

        # Update parameters along dimension j:
        # βⱼ = V' * Ψⱼ * Σ⁻¹Uᵀzⱼ)
        mul!(buffer, U', zⱼ)
        ldiv!(Σ, buffer)
        lmul!(Ψⱼ, buffer)
        mul!(βⱼ, V, buffer)
    end

    return nothing
end