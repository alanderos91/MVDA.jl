"""
Solve via projected gradient descent on a quadratic surrogate.
"""
struct PGD <: AbstractMMAlg end

# Initialize data structures.
function __mm_init__(::PGD, projection_type, problem::MVDAProblem, ::Nothing)
    @unpack coeff = problem
    A = design_matrix(problem)
    n, p, c = probdims(problem)
    nd = vertex_dimension(problem.encoding)
    T = floattype(problem)
    nparams = ifelse(problem.kernel isa Nothing, p, n)

    # projection
    projection = make_projection(projection_type, nparams, c)

    # constants
    Abar = vec(mean(A, dims=1))

    # worker arrays
    Z = fill!(similar(A, n, nd), zero(T))
    buffer = fill!(similar(A, nd, nd), zero(T))

    return (;projection=projection, Abar=Abar, Z=Z, buffer=buffer)
end

# Assume extras has the correct data structures.
__mm_init__(::PGD, projection_type, problem::MVDAProblem, extras) = extras

# Update data structures due to changing ρ.
__mm_update_rho__(::PGD, problem::MVDAProblem, extras, lambda, rho) = nothing

# Update data structures due to changing λ. 
__mm_update_lambda__(::PGD, problem::MVDAProblem, extras, lambda, rho) = nothing

# Apply one update in distance penalized problem.
function __mm_iterate__(::PGD, problem::MVDAProblem, extras, epsilon, lambda, rho, k)
    n, p, _ = probsizes(problem)
    T = floattype(problem)
    alpha, beta = T(1/n), T(lambda/p)
    evaluate_residuals!(problem, extras, epsilon, true, false)
    evaluate_gradient!(problem, lambda)
    __steepest_descent__(problem, extras, alpha, beta)
    extras.projection(problem.coeff.slope, k)
    
    return nothing
end

# Apply one update in regularized problem; same as SGD since there are no constraints.
function __mm_iterate__(::PGD, problem::MVDAProblem, extras, epsilon, lambda)
    n, p, _ = probsizes(problem)
    T = floattype(problem)
    alpha, beta = T(1/n), T(lambda/p)
    evaluate_residuals!(problem, extras, epsilon, true, false)
    evaluate_gradient!(problem, lambda)
    __steepest_descent__(problem, extras, alpha, beta)

    return nothing
end
