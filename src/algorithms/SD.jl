"""
Solve least squares problem via steepest descent.
"""
struct SD <: AbstractMMAlg end

# Initialize data structures.
function __mm_init__(::SD, problem, ::Nothing)
    @unpack X, coeff = problem
    n, p, _ = probdims(problem)
    nparams = ifelse(problem.kernel isa Nothing, p, n)

    # residuals subroutine requires an object named Z; need to fix
    Z = nothing

    return (;
    projection=StructuredL0Projection(nparams), Z=Z,
    )
end

# Check for data structure allocations; otherwise initialize.
function __mm_init__(::SD, problem, extras)
    if :projection in keys(extras) && :Z in keys(extras) # TODO
        return extras
    else
        __mm_init__(SD(), problem, nothing)
    end
end

# Update data structures due to change in model subsets, k.
__mm_update_sparsity__(::SD, problem, ϵ, ρ, k, extras) = nothing

# Update data structures due to changing ρ.
__mm_update_rho__(::SD, problem, ϵ, ρ, k, extras) = nothing

# Update data structures due to changing λ.
__mm_update_lambda__(::SD, problem, ϵ, λ, extras) = nothing

# Apply one update.
function __mm_iterate__(::SD, problem, ϵ, ρ, k, extras)
    @unpack coeff, proj, grad, res = problem
    @unpack projection = extras
    X = get_design_matrix(problem)
    n, _, _ = probdims(problem)
    T = floattype(problem)
    a², b² = 1/n, ρ

    # Project and then evaluate gradient.
    apply_projection(projection, problem, k)
    __evaluate_residuals__(problem, ϵ, extras, true, true, false)
    __evaluate_gradient__(problem, ρ, extras)

    # Find optimal step size
    B, G, XG = coeff.all, grad.all, res.main.all
    mul!(XG, X, G)
    C1 = dot(G, G)
    C2 = dot(XG, XG)
    t = C1 / (a²*C2 + b²*C1 + eps())

    # Move in the direction of steepest descent.
    axpy!(-t, G, B)

    return nothing
end


# Apply one update in regularized problem.
function __reg_iterate__(::SD, problem, ϵ, λ, extras)
    @unpack coeff, grad, res = problem
    X = get_design_matrix(problem)
    n, _, _ = probdims(problem)
    a², b² = 1/n, λ

    # Evaluate the gradient using residuals.
    __evaluate_residuals__(problem, ϵ, extras, true, false, false)
    __evaluate_reg_gradient__(problem, λ, extras)

    # Find optimal step size
    B, G, XG = coeff.all, grad.all, res.main.all
    mul!(XG, X, G)
    C1 = dot(G, G)
    C2 = dot(XG, XG)
    t = C1 / (a²*C2 + b²*C1 + eps())

    # Move in the direction of steepest descent.
    axpy!(-t, G, B)

    return nothing
end