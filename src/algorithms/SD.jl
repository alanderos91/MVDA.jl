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
    projection = make_projection(projection_type, rng, nparams, nd)

    # constants
    Abar = vec(mean(A, dims=1))

    # worker arrays
    Z = fill!(similar(A, n, nd), zero(T))
    buffer = fill!(similar(A, nd, nd), zero(T))

    return (;projection=projection, Abar=Abar, Z=Z, buffer=buffer)
end

# Assume extras has the correct data structures.
__mm_init__(::SD, (projection_type, rng), problem::MVDAProblem, extras) = extras

# Update data structures due to changing hyperparameters
__mm_update_datastructures__(::SD, f::AbstractVDAModel, problem, extras, hparams) = nothing

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

function __mm_iterate__(::SD, f::PenalizedObjective{SqEpsilonLoss,PENALTY},
    problem::MVDAProblem, extras, hparams) where PENALTY <: Union{RidgePenalty,SqDistPenalty}
    #
    n, _, _ = probsizes(problem)
    T = floattype(problem)    
    scale_factor = get_scale_factor(f, problem, extras, hparams)

    if PENALTY <: RidgePenalty
        scaled_lambda = hparams.lambda*scale_factor
        alpha, beta = T(1/n), T(scaled_lambda)
    else
        scaled_rho = hparams.rho*scale_factor
        alpha, beta = T(1/n), T(scaled_rho)
    end

    evaluate_model!(f, problem, extras, hparams)
    __steepest_descent__(problem, extras, alpha, beta)

    return nothing
end

# function __mm_iterate__(::SD, f::PenalizedObjective{SqEpsilonLoss,LassoPenalty},
#     problem::MVDAProblem, extras, hparams)
#     #
#     B = problem.coeff
#     G = deepcopy(problem.grad)
#     T = floattype(problem)

#     old_state = evaluate_model!(f, problem, extras, hparams)
#     t, steps = T(1.0), 0
#     not_decreased = true
#     while not_decreased && steps < 64
#         axpy!(-t, G.slope, B.slope)
#         axpy!(-t, G.intercept, B.intercept)

#         new_state = evaluate_model!(f, problem, extras, hparams)
#         not_decreased = new_state.objective >= old_state.objective

#         if not_decreased
#             axpy!(t, G.slope, B.slope)
#             axpy!(t, G.intercept, B.intercept)
#             t = t / 1.2
#         end
#         steps += 1
#     end
#     @assert !not_decreased
    
#     return nothing
# end
