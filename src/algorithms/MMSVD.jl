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
__mm_update_rho__(::MMSVD, problem::MVDAProblem, ϵ, ρ, k, extras) = update_diagonal(problem, ρ, extras)

# Update data structures due to changing λ. 
__mm_update_lambda__(::MMSVD, problem::MVDAProblem, ϵ, λ, extras) = update_diagonal(problem, λ, extras)

function update_diagonal(problem::MVDAProblem, λ, extras)
    @unpack s, Ψ = extras
    n, _, _ = probdims(problem)
    a², b² = 1/n, λ

    # Update the diagonal matrices Ψ = (a² Σ²) / (a² Σ² + b² I).
    for i in eachindex(Ψ.diag)
        sᵢ² = s[i]^2
        Ψ.diag[i] = a² * sᵢ² / (a² * sᵢ² + b²)
    end

    return nothing
end

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

# Apply one update in reguarlized problem.
function __reg_iterate__(::MMSVD, problem::MVDAProblem, ϵ, λ, extras)
    @unpack intercept, coeff = problem
    @unpack buffer = extras
    @unpack Z, Ψ, U, s, V = extras
    Σ = Diagonal(s)

    # need to compute Z via residuals...
    __evaluate_residuals__(problem, ϵ, extras, true, false, true)

    # Update parameters: B = V * Ψ * Σ⁻¹ * Uᵀ * Z
    B = coeff.all
    mul!(buffer, U', Z)
    ldiv!(Σ, buffer)
    lmul!(Ψ, buffer)
    mul!(B, V, buffer)

    return nothing
end
