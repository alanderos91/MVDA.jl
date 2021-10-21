"""
Solve least squares problem via steepest descent.
"""
struct SD <: AbstractMMAlg end

# Initialize data structures.
function __mm_init__(::SD, problem, ::Nothing)
    @unpack X, coeff = problem
    n, p, _ = probdims(problem)
    T = floattype(problem)
    nparams = ifelse(problem isa MVDAProblem, p, n)

    # residuals subroutine requires an object named Z; need to fix
    Z = nothing

    return (;
    projection=StructuredL0Projection(nparams), Z=Z,
    )
end

# Check for data structure allocations; otherwise initialize.
function __mm_init__(::SD, problem, extras)
    if projection in keys(extras) && :Z in keys(extras) # TODO
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

    # Project and then evaluate gradient.
    apply_projection(projection, problem, k)
    __evaluate_residuals__(problem, ϵ, extras, true, true, false)
    __evaluate_gradient__(problem, ρ, extras)

    # Find optimal step size
    C1 = zero(T)
    C2 = zero(T)
    a² = 1/n
    b² = ρ
    for j in eachindex(grad.dim)
        gⱼ = grad.dim[j]
        Xgⱼ = res.main.dim[j]
        mul!(Xgⱼ, X, gⱼ)
        normgⱼ² = dot(gⱼ, gⱼ)
        normXgⱼ² = dot(Xgⱼ, Xgⱼ)

        C1 += normgⱼ²
        C2 += normXgⱼ²
    end
    t = C1 / (a²*C2 + b²*C1)

    # Move in the direction of steepest descent.
    axpy!(-t, grad.all, coeff.all)

    return nothing
end


# Apply one update in regularized version.
function __mm_iterate__(::SD, problem, ϵ, λ, extras)
    @unpack coeff, proj, grad, res = problem
    X = get_design_matrix(problem)
    n, _, _ = probdims(problem)
    T = floattype(problem)

    # Evaluate gradient.
    __evaluate_residuals__(problem, ϵ, extras, true, false, false)
    __evaluate_gradient_reg__(problem, λ, extras)

    # Find optimal step size
    C1 = zero(T)
    C2 = zero(T)
    a² = 1/n
    b² = λ
    for j in eachindex(grad.dim)
        gⱼ = grad.dim[j]
        Xgⱼ = res.main.dim[j]
        mul!(Xgⱼ, X, gⱼ)
        normgⱼ² = dot(gⱼ, gⱼ)
        normXgⱼ² = dot(Xgⱼ, Xgⱼ)

        C1 += normgⱼ²
        C2 += normXgⱼ²
    end
    t = C1 / (a²*C2 + b²*C1)

    # Move in the direction of steepest descent.
    axpy!(-t, grad.all, coeff.all)

    return nothing
end
