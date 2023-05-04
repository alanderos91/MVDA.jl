function solve!(f::AbstractVDAModel, args...; kwargs...)
    solve_unconstrained!(f, args...; kwargs...)
end

function solve!(f::PenalizedObjective{LOSS,SqDistPenalty}, args...; kwargs...) where LOSS
    solve_constrained!(f, args...; kwargs...)
end

"""
```solve!(algorithm, problem, ϵ, λ, [_extras_]; [maxiter=10^3], [gtol=1e-6], [nesterov=10], [verbose=false])```

Fit an ℓ₂ regularized VDA model with hyperparameters (ϵ, λ).
"""
function solve_unconstrained!(f::AbstractVDAModel, algorithm::AbstractMMAlg, problem::MVDAProblem, hparams,
    _extras_::T=nothing;
    maxiter::Int=DEFAULT_MAXITER,
    gtol::Real=DEFAULT_GTOL,
    nesterov::Int=DEFAULT_NESTEROV,
    callback::F=DEFAULT_CALLBACK,
    kwargs...,
    ) where {T,F}
    # Check for missing data structures.
    extras = __mm_init__(algorithm, (Nothing, nothing), problem, _extras_)
    floatT = floattype(problem)

    # Get problem info and extra data structures.
    @unpack coeff, coeff_prev, coeff_proj = problem

    # Fix hyperparameters.
    hyperparams = (; hparams..., rho=zero(floatT))

    # Update data structures due to hyperparameters.
    __mm_update_datastructures__(algorithm, f, problem, extras, hparams)

    # Check initial values for loss, objective, distance, and norm of gradient.
    state = evaluate_model!(f, problem, extras, hyperparams)
    callback((0, state), problem, hyperparams)
    old = state.objective

    # Initialize iteration counts.
    iters = 0
    nesterov_iter = 1

    if state.gradient < gtol
        return ((iters, state), zero(floatT))
    end

    for iter in 1:maxiter
        iters += 1

        # Apply an algorithm map to update estimates.
        __mm_iterate__(algorithm, f, problem, extras, hyperparams)

        # Update loss, objective, and gradient.
        state = evaluate_model!(f, problem, extras, hyperparams)
        callback((iter, state), problem, hyperparams)

        # Assess convergence.
        obj = state.objective
        if state.gradient < gtol
            break
        elseif iter < maxiter
            needs_reset = iter < nesterov || obj > old
            nesterov_iter = nesterov_acceleration!(coeff, coeff_prev, nesterov_iter, needs_reset)
            old = obj
        end
    end
    copyto!(coeff_proj.slope, coeff.slope)
    copyto!(coeff_proj.intercept, coeff.intercept)

    return ((iters, state), zero(floatT))
end

"""
    solve_constrained!(algorithm, problem, ϵ, λ, s, [extras], [update_extras]; kwargs...)

Fit a sparse VDA model with hyperparameters (ϵ, λ, s) by solving a sequence of unconstrained problems.

!!! Note
    The caller should specify whether to update data structures depending on `s` and `ρ` using `update_extras[1]` and `update_extras[2]`, respectively.

    Convergence is determined based on the rule `dist < dtol || abs(dist - old) < rtol * (1 + old)`, where `dist` is the squared distance and `dtol` and `rtol` are tolerance parameters.

!!! Tip
    The `extras` argument can be constructed using `extras = __mm_init__(algorithm, (rng, projection_type), problem, nothing)`.

# Keyword Arguments

- `nouter`: The number of outer iterations; i.e. the maximum number of `ρ` values to use in annealing (default=`100`).
- `dtol`: An absolute tolerance parameter for the squared distance (default=`1e-6`).
- `rtol`: A relative tolerance parameter for the squared distance (default=`1e-6`).
- `rho_init`: The initial value for `ρ` (default=1.0).
- `rho_max`: The maximum value for `ρ` (default=1e8).
- `rhof`: A function `rhof(ρ, iter, rho_max)` used to determine the next value for `ρ` in the annealing sequence. The default multiplies `ρ` by `1.2`.
- `verbose`: Print convergence information (default=`false`).
- `cb`: A callback function for extending functionality.

See also: [`MVDA.solve_unconstrained!`](@ref) for additional keyword arguments applied at the annealing step.
"""
function solve_constrained!(f::AbstractVDAModel, algorithm::AbstractMMAlg, problem::MVDAProblem, hparams,
    _extras_::T=nothing;
    projection_type::Type=HomogeneousL0Projection,
    maxrhov::Int=DEFAULT_MAXRHOV,
    dtol::Real=DEFAULT_DTOL,
    rtol::Real=DEFAULT_RTOL,
    rho_init::Real=DEFAULT_RHO_INIT,
    rho_max::Real=DEFAULT_RHO_MAX,
    rhof::Function=DEFAULT_ANNEALING,
    callback::F=DEFAULT_CALLBACK,
    rng::AbstractRNG=Random.GLOBAL_RNG,
    kwargs...) where{T,F}
    # Check for missing data structures.
    extras = __mm_init__(algorithm, (projection_type, rng), problem, _extras_)

    # Get problem info and extra data structures.
    @unpack coeff, coeff_prev, coeff_proj = problem
    @unpack projection = extras

    # Initialize ρ and iteration count.
    rho, iters = rho_init, 0
    hyperparams = (; hparams..., rho=rho)
    
    # Update data structures due to hyperparameters.
    __mm_update_datastructures__(algorithm, f, problem, extras, hyperparams)

    # Check initial values for loss, objective, distance, and norm of gradient.
    state = evaluate_model!(f, problem, extras, hyperparams)
    callback((0, state), problem, hyperparams)
    old = state.distance

    for iter in 1:maxrhov
        # Solve minimization problem for fixed rho.
        ((inner_iters, state), _) = solve_unconstrained!(f, algorithm, problem, hyperparams, extras;
            callback=callback,
            kwargs...,
        )

        # Update total iteration count.
        iters += inner_iters

        # Check for convergence to constrained solution.
        dist = state.distance
        if dist < dtol || abs(dist - old) < rtol * (1 + old)
            break
        else
          old = dist
        end
                
        # Update according to annealing schedule.
        if iter < maxrhov
            rho = rhof(rho, iter, rho_max)
            hyperparams = (; hyperparams..., rho=rho,)
        end
    end
    
    # Project solution to the constraint set.
    state = evaluate_model!(f, problem, extras, hyperparams)

    return ((iters, state), rho)
end

"""
    solve_unconstrained!(algorithm, problem, ϵ, λ, s, ρ, [extras], [update_extras]; kwargs...)

Fit a distance-penalized VDA model with penalty strength ρ.

!!! Note
    The caller should specify whether to update data structures depending on `s` and `ρ` using `update_extras[1]` and `update_extras[2]`, respectively.

    Convergence is determined based on the rule `gradsq < gtol`, where `gradsq` is squared Euclidean norm of the gradient and `gtol` is a tolerance parameter.

!!! Tip
    The `extras` argument can be constructed using `extras = __mm_init__(algorithm, (projection_type, rng), problem, nothing)`.

# Keyword Arguments

- `ninner`: The maximum number of iterations (default=`10^4`).
- `gtol`: An absoluate tolerance parameter on the squared Euclidean norm of the gradient (default=`1e-6`).
- `nesterov_threshold`: The number of early iterations before applying Nesterov acceleration (default=`10`).
- `verbose`: Print convergence information (default=`false`).
- `cb`: A callback function for extending functionality.
"""
function solve_unconstrained!(f::PenalizedObjective{LOSS,SqDistPenalty}, algorithm::AbstractMMAlg, problem::MVDAProblem, hyperparams,
    _extras_::T=nothing;
    projection_type::Type=HomogeneousL0Projection,
    maxiter::Int=10^4,
    gtol::Real=1e-6,
    nesterov::Int=10,
    callback::F=DEFAULT_CALLBACK,
    rng::AbstractRNG=Random.GLOBAL_RNG,
    ) where {LOSS<:AbstractVDALoss,T,F}
    # Check for missing data structures.
    extras = __mm_init__(algorithm, (projection_type, rng), problem, _extras_)

    # Get problem info and extra data structures.
    @unpack coeff, coeff_prev, coeff_proj = problem
    @unpack projection = extras

    # Update data structures due to hyperparameters.
    __mm_update_datastructures__(algorithm, f, problem, extras, hyperparams)

    # Check initial values for loss, objective, distance, and norm of gradient.
    state = evaluate_model!(f, problem, extras, hyperparams)
    old = state.objective

    # Initialize iteration counts.
    iters = 0
    nesterov_iter = 1
    for iter in 1:maxiter
        iters += 1

        # Apply an algorithm map to update estimates.
        __mm_iterate__(algorithm, f, problem, extras, hyperparams)

        # Update loss, objective, distance, and gradient.
        state = evaluate_model!(f, problem, extras, hyperparams)
        callback((iter, state), problem, hyperparams)

        # Assess convergence.
        obj = state.objective
        if state.gradient < gtol
            break
        elseif iter < maxiter
            needs_reset = iter < nesterov || obj > old
            nesterov_iter = nesterov_acceleration!(coeff, coeff_prev, nesterov_iter, needs_reset)
            old = obj
        end
    end
    # Save parameter estimates in case of warm start.
    copyto!(coeff_prev.slope, coeff.slope)
    copyto!(coeff_prev.intercept, coeff.intercept)

    return ((iters, state), hyperparams.rho)
end

# function solve_constrained!(algorithm::PGD, problem::MVDAProblem, epsilon::Real, lambda::Real, s::Real,
#     _extras_::T=nothing;
#     projection_type::Type=HomogeneousL0Projection,
#     maxiter::Int=10^4,
#     gtol::Real=DEFAULT_GTOL,
#     rtol::Real=DEFAULT_RTOL,
#     rho_init::Real=DEFAULT_RHO_INIT,
#     nesterov::Int=10,
#     callback::F=DEFAULT_CALLBACK,
#     rng::AbstractRNG=Random.GLOBAL_RNG,
#     kwargs...) where{T,F}
#     # Check for missing data structures.
#     extras = __mm_init__(algorithm, (projection_type, rng), problem, _extras_)

#     # Get problem info and extra data structures.
#     @unpack coeff, coeff_prev, coeff_proj = problem
#     @unpack projection = extras

#     # Initialize ρ and iteration count.
#     rho, iters = rho_init, 0

#     # Fix hyperparameters.
#     k = sparsity_to_k(problem, s)
#     hyperparams = (;epsilon=epsilon, lambda=lambda, rho=rho, k=k)
    
#     # Update data structures due to hyperparameters.
#     __mm_update_lambda__(algorithm, problem, extras, lambda, zero(rho))

#     # Check initial values for loss, objective, distance, and norm of gradient.
#     state = evaluate_objective_pgd!(problem, extras, hyperparams, 1.0)
#     callback((0, state), problem, hyperparams)
#     old = state.objective

#     # Initialize iteration counts.
#     iters = 0
#     nesterov_iter = 1
#     for iter in 1:maxiter
#         iters += 1

#         # Apply an algorithm map to update estimates.
#         t = __mm_iterate__(algorithm, problem, extras, hyperparams)

#         # Update loss, objective, distance, and gradient.
#         state = evaluate_objective_pgd!(problem, extras, hyperparams, t)
#         callback((iter, state), problem, hyperparams)

#         # Assess convergence.
#         obj = state.objective
#         maxcoeff = norm(coeff_prev.slope, Inf)
#         if problem.intercept
#             maxcoeff = max(maxcoeff, norm(coeff_prev.slope, Inf))
#         end
#         if state.gradient < gtol || state.gradient < rtol*(1+maxcoeff)
#             break
#         elseif iter < maxiter
#             needs_reset = iter < nesterov || obj > old
#             nesterov_iter = nesterov_acceleration!(coeff, coeff_prev, nesterov_iter, needs_reset)
#             old = obj
#         end
#     end
#     # Save parameter estimates in case of warm start.
#     copyto!(coeff_proj.slope, coeff.slope)
#     copyto!(coeff_proj.intercept, coeff.intercept)
#     copyto!(coeff_prev.slope, coeff.slope)
#     copyto!(coeff_prev.intercept, coeff.intercept)

#     return ((iters, state), rho)
# end
