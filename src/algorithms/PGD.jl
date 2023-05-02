"""
Solve via projected gradient descent on a quadratic surrogate.
"""
struct PGD <: AbstractMMAlg end

# Initialize data structures.
function __mm_init__(::PGD, (projection_type, rng), problem::MVDAProblem, ::Nothing)
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
__mm_init__(::PGD, (projection_type, rng), problem::MVDAProblem, extras) = extras

# Update data structures due to changing ρ.
__mm_update_rho__(::PGD, problem::MVDAProblem, extras, lambda, rho) = nothing

# Update data structures due to changing λ. 
__mm_update_lambda__(::PGD, problem::MVDAProblem, extras, lambda, rho) = nothing

# Apply one update in distance penalized problem.
function __mm_iterate__(::PGD, problem::MVDAProblem, extras, hyperparams)
    @unpack epsilon, lambda, rho, k = hyperparams
    n, p, _ = probsizes(problem)
    T = floattype(problem)
    alpha, beta = T(1/n), T(lambda/p)
    evaluate_residuals!(problem, extras, epsilon, true, false)
    evaluate_gradient!(problem, lambda)
    t = __steepest_descent__(problem, extras, alpha, beta)
    extras.projection(problem.coeff.slope, k)
    
    return t
end

# Apply one update in regularized problem; same as SGD since there are no constraints.
function __mm_iterate_reg__(::PGD, problem::MVDAProblem, extras, hyperparams)
    @unpack epsilon, lambda = hyperparams
    n, p, _ = probsizes(problem)
    T = floattype(problem)
    alpha, beta = T(1/n), T(lambda/p)
    evaluate_residuals!(problem, extras, epsilon, true, false)
    evaluate_gradient!(problem, lambda)
    __steepest_descent__(problem, extras, alpha, beta)

    return nothing
end

"""
Evaluate loss + norm of gradient mapping in PGD.
"""
function evaluate_objective_pgd!(problem::MVDAProblem, extras, hyperparams, t)
    @unpack epsilon, lambda = hyperparams
    @unpack coeff, coeff_prev, res, grad = problem
    n, p, _ = probsizes(problem)

    evaluate_residuals!(problem, extras, epsilon, true, false)

    # Evaluate gradient mapping 1/t*(Bₘ - Bₘ₊₁)
    grad.slope .= coeff_prev.slope - coeff.slope
    gradsq = norm(grad.slope, Inf)
    if problem.intercept
        grad.intercept .= coeff_prev.intercept - coeff.intercept
        gradsq = max(gradsq, norm(grad.intercept, Inf))
    end

    B, R = coeff.slope, res.loss
    risk = 1//n * dot(R, R)
    penalty = dot(B, B)
    loss = 1//2 * (risk + lambda/p*penalty)
    distsq = zero(loss)
    obj = loss
    gradsq *= gradsq / (t^2)

    return __eval_result__(risk, loss, obj, penalty, distsq, gradsq)
end
