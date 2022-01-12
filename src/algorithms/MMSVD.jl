"""
Solve the least squares problem using a SVD of the design matrix.
"""
struct MMSVD <: AbstractMMAlg end

# Initialize data structures.
function __mm_init__(::MMSVD, problem::MVDAProblem, ::Nothing)
    @unpack coeff = problem
    X = get_design_matrix(problem)
    n, p, c = probdims(problem)
    T = floattype(problem)
    nparams = ifelse(problem.kernel isa Nothing, p, n)

    # thin SVD of X
    U, s, V = __svd_wrapper__(X)
    r = length(s) # number of nonzero singular values

    # worker arrays
    Z = Matrix{T}(undef, n, c-1)
    buffer = Matrix{T}(undef, r, c-1)
    
    # diagonal matrices
    Ψ = Diagonal(Vector{T}(undef, r))
    
    return (;
        projection=StructuredL0Projection(nparams),
        U=U, s=s, V=V,
        Z=Z, Ψ=Ψ,
        buffer=buffer,
    )
end

# Check for data structure allocations; otherwise initialize.
function __mm_init__(::MMSVD, problem::MVDAProblem, extras)
    if :projection in keys(extras) && :buffer in keys(extras) # TODO
        return extras
    else
        __mm_init__(MMSVD(), problem, nothing)
    end
end

# Update data structures due to change in model size, k.
__mm_update_sparsity__(::MMSVD, problem::MVDAProblem, ϵ, ρ, k, extras) = nothing

# Update data structures due to changing ρ.
function __mm_update_rho__(::MMSVD, problem::MVDAProblem, ϵ, ρ, k, extras)
    @unpack s, Ψ = extras
    n, _, _ = probdims(problem)
    a², b² = 1/n, ρ

    # Update the diagonal matrices Ψ = (a² Σ²) / (a² Σ² + b² I).
    for i in eachindex(Ψ.diag)
        sᵢ² = s[i]^2
        Ψ.diag[i] = a² * sᵢ² / (a² * sᵢ² + b²)
    end

    return nothing
end

# # Update data structures due to changing λ.
# function __mm_update_lambda__(::MMSVD, problem::MVDAProblem, ϵ, λ, extras)
#     @unpack s, Ψ = extras
#     n, _, _ = probdims(problem)
#     a² = 1 / n

#     # Update the diagonal matrices Ψⱼ = (a² Σ²) / (a² Σ² + b² I).
#     for j in eachindex(Ψ)
#         Ψⱼ = Ψ[j]
#         b² = λ
#         for i in eachindex(Ψⱼ.diag)
#             sᵢ² = s[i]^2
#             Ψⱼ.diag[i] = a² * sᵢ² / (a² * sᵢ² + b²)
#         end
#     end

#     return nothing
# end

# Apply one update.
function __mm_iterate__(::MMSVD, problem::MVDAProblem, ϵ, ρ, k, extras)
    @unpack intercept, coeff, proj = problem
    @unpack buffer, projection = extras
    @unpack Z, Ψ, U, s, V = extras
    Σ = Diagonal(s)
    T = floattype(problem)

    # need to compute Z via residuals...
    apply_projection(projection, problem, k)
    __evaluate_residuals__(problem, ϵ, extras, true, false, true)

    # Update parameters:B = P + V * Ψ * (Σ⁻¹UᵀZ - VᵀP) 
    B = coeff.all
    P = proj.all
    mul!(buffer, U', Z)
    ldiv!(Σ, buffer)
    mul!(buffer, V', P, -one(T), one(T))
    lmul!(Ψ, buffer)
    mul!(B, V, buffer)
    axpy!(one(T), P, B)

    return nothing
end

# # Apply one update in regularized version.
# function __mm_iterate__(::MMSVD, problem::MVDAProblem, ϵ, λ, extras)
#     @unpack intercept, coeff, proj = problem
#     @unpack buffer = extras
#     @unpack Z, Ψ, U, s, V = extras
#     Σ = Diagonal(s)

#     # need to compute Z via residuals...
#     __evaluate_residuals__(problem, ϵ, extras, true, false, true)

#     for j in eachindex(coeff.dim)
#         # Rename relevant arrays/views
#         βⱼ = coeff.dim[j]
#         zⱼ = view(Z, :, j)
#         Ψⱼ = Ψ[j]

#         # Update parameters along dimension j:
#         # βⱼ = V' * Ψⱼ * Σ⁻¹Uᵀzⱼ)
#         mul!(buffer, U', zⱼ)
#         ldiv!(Σ, buffer)
#         lmul!(Ψⱼ, buffer)
#         mul!(βⱼ, V, buffer)
#     end

#     return nothing
# end