"""
Solve via projected gradient descent.
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
    projection = make_projection(projection_type, rng, nparams, nd)

    # constants
    Abar = vec(mean(A, dims=1))

    # worker arrays
    Z = fill!(similar(A, n, nd), zero(T))
    buffer = fill!(similar(A, nd, nd), zero(T))

    return (;projection=projection, Abar=Abar, Z=Z, buffer=buffer)
end

# Assume extras has the correct data structures.
__mm_init__(::PGD, (projection_type, rng), problem::MVDAProblem, extras) = extras

__mm_update_datastructures__(::PGD, ::AbstractVDAModel, problem, extras, hparams) = nothing

function __mm_iterate__(::PGD, f::UnpenalizedObjective{LOSS},
    problem::MVDAProblem, extras, hparams) where LOSS <: AbstractVDALoss
    #
    n, _, _ = probsizes(problem)
    T = floattype(problem)    

    evaluate_model!(f, problem, extras, hparams)
    t = __steepest_descent__(problem, extras, T(1/n), zero(T))
    apply_projection(problem, extras, hparams, true) # in-place

    return t
end
