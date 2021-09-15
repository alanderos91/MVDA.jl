"""
Solve least squares problem via steepest descent.
"""
struct SD <: AbstractMMAlg end

# Initialize data structures.
function __mm_init__(::SD, problem, ::Nothing)
    @unpack X, coeff = problem
    n, p, c = probdims(problem)
    T = floattype(problem)

    # residuals subroutine requires an object named Z; need to fix
    Z = nothing

    # diagonal matrix for prefactors
    D = Diagonal(Vector{T}(undef, c-1))

    return (;
        apply_projection=ApplyStructuredL0Projection(p), D=D, Z=Z,
    )
end

# Check for data structure allocations; otherwise initialize.
function __mm_init__(::SD, problem, extras)
    if :apply_projection in keys(extras) && :D in keys(extras) # TODO
        return extras
    else
        __mm_init__(SD(), problem, nothing)
    end
end

# Update data structures due to change in model subsets, k.
function __mm_update_sparsity__(::SD, problem, ϵ, ρ, k, extras)
    @unpack D = extras
    n, p, c = probdims(problem)

    # Update scaling factors on distance penalty, Dⱼⱼ = 1 / √( (c-1) * (p-kⱼ+1) )
    @inbounds for j in eachindex(D.diag)
        # D.diag[j] = 1 / sqrt( (c-1) * (p-k[j]+1) )
        D.diag[j] = 1 / sqrt(p)
    end

    return nothing
end

# Update data structures due to changing ρ.
__mm_update_rho__(::SD, problem, ϵ, ρ, k, extras) = nothing

# Apply one update.
function __mm_iterate__(::SD, problem, ϵ, ρ, k, extras)
    @unpack X, intercept, coeff, proj, grad, res = problem
    @unpack apply_projection, D = extras
    n, p, c = probdims(problem)
    T = floattype(problem)

    # Project and then evaluate gradient.
    copyto!(proj.all, coeff.all)
    apply_projection(view(proj.all, 1:p, :), k)
    __evaluate_residuals__(problem, ϵ, extras, true, true, false)
    __evaluate_gradient__(problem, ρ, extras)

    # Find optimal step size

    A = zero(T)
    B = zero(T)
    a² = 1/n
    for j in axes(B, 2)
        gⱼ = grad.dim[j]
        Xgⱼ = res.main.dim[j]
        bⱼ² = ρ * D[j,j]^2
        mul!(Xgⱼ, X, gⱼ)
        normgⱼ² = dot(gⱼ, gⱼ)
        normXgⱼ² = dot(Xgⱼ, Xgⱼ)

        A += normgⱼ²
        B += a² * normXgⱼ² + bⱼ² * normgⱼ²
    end
    t = A / B

    # Move in the direction of steepest descent.
    axpy!(-t, grad.all, coeff.all)

    return nothing
end
