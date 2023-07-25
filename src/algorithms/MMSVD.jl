"""
Solve via a sequence of linear system solves. Uses SVD of design matrix.
"""
struct MMSVD <: AbstractMMAlg end

# Initialize data structures.
function __mm_init__(::MMSVD, (projection_type, rng), problem::MVDAProblem, ::Nothing)
    @unpack coeff = problem
    A = design_matrix(problem)
    n, p, c = probdims(problem)
    nd = vertex_dimension(problem.encoding)
    T = floattype(problem)
    nparams = ifelse(problem.kernel isa Nothing, p, n)

    # projection
    projection = make_projection(projection_type, rng, nparams, nd)

    # thin SVD of A
    F = __compute_svd__(A)
    r = length(F.S)

    # constants
    Abar = vec(mean(A, dims=1))
    
    # worker arrays
    Z = fill!(similar(A, n, nd), zero(T))
    buffer = fill!(similar(A, r, nd), zero(T))
    Psi = Diagonal(fill!(similar(A, r), 0))
    
    return (;
        projection=projection,
        U=F.U, s=F.S, V=Matrix(F.V),
        Abar=Abar, Z=Z, buffer=buffer, Psi=Psi,
    )
end

# Assume extras has the correct data structures.
__mm_init__(::MMSVD, (projection_type, rng), problem::MVDAProblem, extras) = extras

function __mm_update_datastructures__(::MMSVD, f::PenalizedObjective{SqEpsilonLoss,RidgePenalty},
    problem, extras, hparams)
    #
    n, _, _ = probsizes(problem)
    T = floattype(problem)
    scale_factor = get_scale_factor(f, problem, extras, hparams)
    scaled_lambda = hparams.lambda*scale_factor
    update_diagonal(extras, T(1/n), T(scaled_lambda))
end

function __mm_update_datastructures__(::MMSVD, f::PenalizedObjective{SqEpsilonLoss,SqDistPenalty},
    problem, extras, hparams)
    n, _, _ = probsizes(problem)
    T = floattype(problem)
    scale_factor = get_scale_factor(f, problem, extras, hparams)
    scaled_rho = hparams.rho*scale_factor
    update_diagonal(extras, T(1/n), T(scaled_rho))
end

function __mm_update_datastructures__(::MMSVD, f::UnpenalizedObjective{SqEpsilonLoss},
    problem, extras, hparams)
    @unpack s, Psi = extras
    n, _, _ = probsizes(problem)
    for i in eachindex(Psi.diag)
        Psi.diag[i] = n / (s[i]^2)
    end
    return nothing
end

function update_diagonal(extras, alpha, beta)
    @unpack s, Psi = extras
    for i in eachindex(Psi.diag)
        s2 = s[i]^2
        Psi.diag[i] = alpha*s2 / (alpha*s2 + beta)
    end
    return nothing
end

# Solves H*B = C, where H = (γ*A'A + β*I), using thin SVD of A.
# The SVD makes it so that H⁻¹ = β⁻¹ * [I - V Ψ Vᵀ] and γ is absorved in Psi.
function __apply_inverse__!(B, Hinv, C, buffer, alpha::Real=zero(eltype(B)))
    beta, V, Psi = Hinv
    if iszero(alpha)    # compute B = H⁻¹ C
        copyto!(B, C)
        BLAS.scal!(1/beta, B)
        alpha = one(beta)
    else                # compute B = B + α * H⁻¹ C
        BLAS.axpy!(alpha/beta, C, B)
    end

    # accumulate Ψ * Vᵀ * C
    BLAS.gemm!('T', 'N', one(beta), V, C, zero(beta), buffer)
    lmul!(Psi, buffer)

    # complete the product with a 5-arg mul
    BLAS.gemm!('N', 'N', -alpha/beta, V, buffer, one(beta), B)

    return nothing
end

function __apply_pseudoinverse__!(B, Hpinv, C, buffer, alpha::Real=zero(eltype(B)))
    V, Psi, _ = Hpinv

    # Determine whether we are accumulating or storing result in B
    if iszero(alpha) # storing
        fill!(B, zero(alpha))
        alpha = one(alpha)
    end# alpha != 0 implies accumulating

    # accumulate Ψ * Vᵀ * C
    BLAS.gemm!('T', 'N', one(alpha), V, C, zero(alpha), buffer)
    lmul!(Psi, buffer)

    # complete the product with a 5-arg mul
    BLAS.gemm!('N', 'N', alpha, V, buffer, one(alpha), B)

    return nothing
end

function __inverse_quadratic_form__(Hinv, x, buffer)
    beta, V, Psi = Hinv
    T = eltype(buffer)
    BLAS.gemv!('T', one(T), V, x, zero(T), buffer)
    for i in eachindex(buffer)
        buffer[i] = sqrt(Psi.diag[i]) * buffer[i]
    end

    return 1/beta * (BLAS.dot(x, x) - BLAS.dot(buffer, buffer))
end

function __pseudoinverse_quadratic_form__(Hpinv, x, buffer)
    V, Psi, _ = Hpinv
    T = eltype(buffer)
    BLAS.gemv!('T', one(T), V, x, zero(T), buffer)
    for i in eachindex(buffer)
        buffer[i] = sqrt(Psi.diag[i]) * buffer[i]
    end

    return BLAS.dot(buffer, buffer)
end

#
#   NOTE: worker arrays must not be aliased with parameters (B, b0)!!!
#
function __linear_solve_SVD__(LHS_and_RHS::Function, apply_inv!, eval_quadratic_form, problem::MVDAProblem, extras)
    @unpack coeff, intercept = problem
    @unpack Abar, Z, buffer = extras

    B, b0 = coeff.slope, coeff.intercept
    T = floattype(problem)    
    Hinv, C = LHS_and_RHS(problem, extras)

    # 1. Solve H*B = RHS.
    apply_inv!(B, Hinv, C, buffer, zero(T))

    # Apply (generalized) Schur complement in H to compute intercept and shift coefficients.
    if intercept
        # 2. Compute (generalized) Schur complement, s = 1 - x̄ᵀH⁻¹x̄
        s = 1 - eval_quadratic_form(Hinv, Abar, view(buffer, :, 1))
        # if s == 0 error("Case not implemented") end
        s = ifelse(iszero(s), one(s), s)
        
        # 3. Compute new intercept, b₀ = s⁻¹[z̄ - Bᵀx̄]
        mean!(b0, Z')
        BLAS.gemv!('T', -T(1/s), B, Abar, T(1/s), b0)

        # 4. Compute new slopes, B = B - H⁻¹(x̄*b₀ᵀ)
        fill!(C, zero(T))
        BLAS.ger!(one(T), Abar, b0, C)
        apply_inv!(B, Hinv, C, buffer, -one(T))
    end

    return nothing
end

# Apply one update.
function __mm_iterate__(::MMSVD, f::PenalizedObjective{SqEpsilonLoss,PENALTY},
    problem::MVDAProblem, extras, hparams) where PENALTY <: Union{RidgePenalty,SqDistPenalty}
    #
    n, _, _ = probsizes(problem)
    T = floattype(problem)
    scale_factor = get_scale_factor(f, problem, extras, hparams)

    if PENALTY <: RidgePenalty
        scaled_lambda = hparams.lambda * scale_factor
        c1, c2, c3 = T(1/n), T(scaled_lambda), zero(T)
    else
        scaled_rho = hparams.rho * scale_factor
        c1, c2, c3 = T(1/n), T(scaled_rho), T(scaled_rho)
    end

    f = let c1=c1, c2=c2, c3=c3
        function(problem, extras)
            A = design_matrix(problem)

            # LHS: H = γ*A'A + β*I; pass as (β, V, Ψ) which computes H⁻¹ = β⁻¹[I - V Ψ Vᵀ]
            H = (c2, extras.V, extras.Psi)

            # RHS: C = 1/n*AᵀZₘ
            C = problem.coeff_proj.slope
            BLAS.gemm!('T', 'N', c1, A, extras.Z, c3, C)
            return H, C
        end
    end

    apply_projection(problem, extras, hparams, false)
    evaluate_residuals!(problem, extras, hparams.epsilon, true, false)
    __linear_solve_SVD__(f, __apply_inverse__!, __inverse_quadratic_form__, problem, extras)

    return nothing
end

function __mm_iterate__(::MMSVD, f::UnpenalizedObjective{SqEpsilonLoss},
    problem::MVDAProblem, extras, hparams)
    #
    @unpack coeff, intercept = problem

    n, _, _ = probsizes(problem)
    T = floattype(problem)
    c1 = T(1/n)
    f = let c1=c1
        function(problem, extras)
            A = design_matrix(problem)

            # LHS: H = A'A; pass as (V, Ψ) to represent H⁻¹ = V Ψ Vᵀ
            H = (extras.V, extras.Psi, extras.U)

            # RHS: C = 1/n*AᵀZₘ
            C = problem.coeff_proj.slope
            BLAS.gemm!('T', 'N', c1, A, extras.Z, zero(T), C)
            return H, C
        end
    end
    evaluate_residuals!(problem, extras, hparams.epsilon, true, false)
    __linear_solve_SVD__(f, __apply_pseudoinverse__!, __pseudoinverse_quadratic_form__, problem, extras)

    return nothing
end
