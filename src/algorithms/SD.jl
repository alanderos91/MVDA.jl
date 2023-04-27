"""
Solve via steepest descent on a quadratic surrogate.
"""
struct SD <: AbstractMMAlg end

# Initialize data structures.
function __mm_init__(::SD, (projection_type, rng), problem::MVDAProblem, ::Nothing)
    @unpack coeff = problem
    A = design_matrix(problem)
    n, p, c = probdims(problem)
    nd = vertex_dimension(problem.encoding)
    T = floattype(problem)
    nparams = ifelse(problem.kernel isa Nothing, p, n)

    # projection
    projection = make_projection(projection_type, rng, nparams, c)

    # constants
    Abar = vec(mean(A, dims=1))

    # worker arrays
    Z = fill!(similar(A, n, nd), zero(T))
    buffer = fill!(similar(A, nd, nd), zero(T))

    return (;projection=projection, Abar=Abar, Z=Z, buffer=buffer)
end

# Assume extras has the correct data structures.
__mm_init__(::SD, (projection_type, rng), problem::MVDAProblem, extras) = extras

# Update data structures due to changing ρ.
__mm_update_rho__(::SD, problem::MVDAProblem, extras, lambda, rho) = nothing

# Update data structures due to changing λ. 
__mm_update_lambda__(::SD, problem::MVDAProblem, extras, lambda, rho) = nothing

function __steepest_descent__(problem::MVDAProblem, extras, alpha, beta)
    @unpack coeff, grad, res, intercept = problem
    @unpack Abar, buffer = extras
    A, G, g0 = design_matrix(problem), grad.slope, grad.intercept
    AG, C, CtG = res.loss, res.dist, buffer # reuse residuals as buffers
    T = floattype(problem)

    # Find optimal step size
    BLAS.gemm!('N', 'N', one(T), A, G, zero(T), AG)
    normGsq = BLAS.dot(G, G)        # tr(G'G)
    normAGsq = BLAS.dot(AG, AG)     # tr(A'G'GA)
    if intercept
        fill!(C, zero(T))
        BLAS.ger!(one(T), Abar, g0, C)
        BLAS.gemm!('T', 'N', one(T), C, G, zero(T), CtG)
        trCtG = tr(CtG)                 # tr(x̄ g₀' G)
        normg0sq = BLAS.dot(g0, g0)     # tr(g₀ g₀')
        numerator = normGsq + normg0sq
        denominator = alpha*normAGsq + beta*normGsq + normg0sq + 2*trCtG
    else
        numerator = normGsq
        denominator = alpha*normAGsq + beta*normGsq
    end
    indeterminate = iszero(numerator) && iszero(denominator)
    t = ifelse(indeterminate, zero(T), numerator / denominator)

    # Move in the direction of steepest descent.
    BLAS.axpy!(-t, G, coeff.slope)
    intercept && BLAS.axpy!(-t, g0, coeff.intercept)

    return t
end

# Apply one update in distance penalized problem.
function __mm_iterate__(::SD, problem::MVDAProblem, extras, epsilon, lambda, rho, k)
    n, p, _ = probsizes(problem)
    T = floattype(problem)
    alpha, beta = T(1/n), T(lambda/p+rho/p)
    apply_projection(extras.projection, problem, k)
    evaluate_residuals!(problem, extras, epsilon, true, true)
    evaluate_gradient!(problem, lambda, rho)
    __steepest_descent__(problem, extras, alpha, beta)

    return nothing
end

# Apply one update in regularized problem.
function __mm_iterate__(::SD, problem::MVDAProblem, extras, epsilon, lambda)
    n, p, _ = probsizes(problem)
    T = floattype(problem)
    alpha, beta = T(1/n), T(lambda/p)
    evaluate_residuals!(problem, extras, epsilon, true, false)
    evaluate_gradient!(problem, lambda)
    __steepest_descent__(problem, extras, alpha, beta)

    return nothing
end
