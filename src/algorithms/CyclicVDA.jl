"""
Solve MVDA problem with cyclic coordinate descent.
"""
struct CyclicVDA <: AbstractMMAlg end


# Initialize data structures.
function __mm_init__(::CyclicVDA, problem, ::Nothing)
    return nothing
end

# Check for data structure allocations; otherwise initialize.
function __mm_init__(::CyclicVDA, problem, extras)
    if :apply_projection in keys(extras) && :buffer in keys(extras) # TODO
        return extras
    else
        __mm_init__(CyclicVDA(), problem, nothing)
    end
end

# Apply one update in regularized version.
function __mm_iterate__(::CyclicVDA, problem, ϵ, δ, λ₁, λ₂, extras)
    @unpack X, intercept, coeff, res = problem
    MAX_NEWTON_STEPS = 100
    MAX_BACKTRACK_STEPS = 5
    
    B = coeff.all
    R = res.main.all

    for j in axes(B, 1) # loop over features
        for k in axes(B, 2) # loop over dimensions in vertex space
            objective, d1objective, d2objective = fetch_objective(problem, j, k, ϵ, δ, λ₁, λ₂)
            new_objective = objective

            if abs(d1objective) ≤ 0 continue end

            for _ in 1:MAX_NEWTON_STEPS # minimize via Newton's method
                β = B[j,k]
                ∇ = -d1objective / d2objective

                # Evaluate the new objective. Use step-halving strategy for line search.
                for _ in 1:MAX_BACKTRACK_STEPS
                    A = β + ∇ - B[j,k]
                    B[j,k] = β + ∇

                    # Update residuals, then reevaluate objective to check for descent property.
                    for i in axes(X, 1)
                        R[i,k] = R[i,k] - A * X[i,j]
                    end
                    new_objective, d1objective, d2objective = fetch_objective(problem, j, k, ϵ, δ, λ₁, λ₂)
                    descent = new_objective ≤ objective

                    if descent break end
                    ∇ = 1//2 * ∇
                end # step-halving
                converged = abs(objective - new_objective) ≤ 1e-4

                if converged break end
            end # Newton
            
            # Reset parameters close to 0 to 0.
            β = B[j,k]
            is_near_zero = abs(β) ≤ 1e-8
            B[j,k] = ifelse(is_near_zero, zero(β), β)
        end # dimensions
    end # features

    return nothing
end

function fetch_objective(problem, j, k, ϵ, δ, λ₁, λ₂) # feature j, dimension k
    @unpack X, coeff, res = problem
    cases, features, _ = probdims(problem)

    # Constants.
    C1 = ϵ - δ
    C2 = ϵ + δ
    C3 = 1 / (16 * δ^3)
    C4 = 1 / (4 * δ^3)
    C5 = 3 / (4 * δ^3)
    μ₁ = cases * λ₁
    μ₂ = cases * λ₂

    # Initialize outputs.
    objective = 0.0
    d1objective = 0.0
    d2objective = 0.0

    R = res.main.all
    B = coeff.all

    # Loop over all cases i.
    for i in axes(R, 1)
        rᵢ = view(R, i, :)
        norm_rᵢ = norm(rᵢ)
        norm_rᵢ² = norm_rᵢ^2

        # Skip the loss associated with case i if its predicted value is within
        # a distance ϵ-δ of its associated vertex.
        if norm_rᵢ < C1 continue end
        
        x = X[i,j]
        rx = rᵢ[k] * x

        # Compute the loss for case i if it lies beyond a distance ϵ+δ of its associated vertex.
        if norm_rᵢ > C2
            objective = objective + norm_rᵢ - ϵ
            s = 1 / norm_rᵢ
            d1objective = d1objective - s*rx
            d2objective = d2objective + s*(x*x - rx*rx/norm_rᵢ²)
        else
            s = norm_rᵢ - C1
            t = C2 - norm_rᵢ
            d²p = C5 * s * t
            s² = s*s
            t = t + δ
            d¹p = C4 * s² * t
            t = t + δ
            s³ = s²*s
            p = C3 * s³ * t
            objective = objective + p
            s = d¹p / norm_rᵢ
            d1objective = d1objective - s*rx
            t = d²p / norm_rᵢ²
            rx² = rx * rx
            d2objective = d2objective + t*rx² + s*(x*x - rx²/norm_rᵢ²)
        end
    end

    # Compute the penalty for the current parameter.
    update_penalty = problem.intercept ? (j != features + 1) : true
    if update_penalty
        βⱼ = view(B, j, :)
        normβⱼ = norm(βⱼ, 2)
        normβⱼ² = normβⱼ^2
        objective = objective + μ₂*normβⱼ + μ₁* norm(βⱼ, 1)

        # If the parameter is parked at 0, then reset the first derivative of the objective
        # so that it either remains at 0 or moves in the correct direction.
        if normβⱼ ≤ 0
            if abs(d1objective) ≤ μ₂ + μ₁
                d1objective = zero(d1objective)
            elseif d1objective + μ₂ + μ₁ ≤ 0
                d1objective = d1objective + μ₂ + μ₁
            else
                d1objective = d1objective - μ₂ - μ₁
            end
        else # Deal with the Euclidean penalty away from the origin.
            s = B[j,k]
            t = s / normβⱼ
            d1objective = d1objective + μ₂*t
            d2objective = d2objective + μ₂*(1-t*t)/normβⱼ

            if abs(d1objective) ≤ μ₁
                d1objective = zero(d1objective)
            elseif d1objective + μ₁ ≤ 0
                d1objective = d1objective + μ₁
            else
                d1objective = d1objective - μ₁
            end
        end
    end

    return objective, d1objective, d2objective
end
