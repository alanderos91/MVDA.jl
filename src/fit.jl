"""
```init!(algorithm, problem, ϵ, λ, [_extras_]; [maxiter=10^3], [gtol=1e-6], [nesterov_threshold=10], [verbose=false])```

Initialize a `problem` with its `λ`-regularized solution.
"""
function StatsBase.fit!(algorithm::AbstractMMAlg, problem::MVDAProblem, epsilon, lambda, _extras_::T=nothing;
    maxiter::Int=DEFAULT_MAXITER,
    gtol::Real=DEFAULT_GTOL,
    nesterov::Int=DEFAULT_NESTEROV,
    callback::F=DEFAULT_CALLBACK,
    kwargs...,
    ) where {T,F}
    # Check for missing data structures.
    extras = __mm_init__(algorithm, problem, _extras_)

    # Get problem info and extra data structures.
    @unpack coeff, coeff_prev, coeff_proj = problem

    # Fix hyperparameters.
    hyperparams = (;epsilon=epsilon, lambda=lambda, rho=zero(lambda), k=size(coeff.slope, 1))

    # Update data structures due to hyperparameters.
    __mm_update_lambda__(algorithm, problem, extras, lambda, zero(lambda))

    # Check initial values for loss, objective, distance, and norm of gradient.
    state = evaluate_objective!(problem, extras, epsilon, lambda)
    callback((0, state), problem, hyperparams)
    old = state.objective

    # Initialize iteration counts.
    iters = 0
    nesterov_iter = 1

    if state.gradient < gtol
        return ((iters, state), zero(lambda))
    end

    for iter in 1:maxiter
        iters += 1

        # Apply an algorithm map to update estimates.
        __mm_iterate__(algorithm, problem, extras, epsilon, lambda)

        # Update loss, objective, and gradient.
        state = evaluate_objective!(problem, extras, epsilon, lambda)
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

    return ((iters, state), zero(lambda))
end

"""
    fit!(algorithm, problem, ϵ, s, [extras], [update_extras]; kwargs...)

Same as `fit_MVDA(algorithm, problem, ϵ, s)`, but with preallocated data structures in `extras`.

!!! Note
    The caller should specify whether to update data structures depending on `s` and `ρ` using `update_extras[1]` and `update_extras[2]`, respectively.

    Convergence is determined based on the rule `dist < dtol || abs(dist - old) < rtol * (1 + old)`, where `dist` is the squared distance and `dtol` and `rtol` are tolerance parameters.

!!! Tip
    The `extras` argument can be constructed using `extras = __mm_init__(algorithm, problem, nothing)`.

# Keyword Arguments

- `nouter`: The number of outer iterations; i.e. the maximum number of `ρ` values to use in annealing (default=`100`).
- `dtol`: An absolute tolerance parameter for the squared distance (default=`1e-6`).
- `rtol`: A relative tolerance parameter for the squared distance (default=`1e-6`).
- `rho_init`: The initial value for `ρ` (default=1.0).
- `rho_max`: The maximum value for `ρ` (default=1e8).
- `rhof`: A function `rhof(ρ, iter, rho_max)` used to determine the next value for `ρ` in the annealing sequence. The default multiplies `ρ` by `1.2`.
- `verbose`: Print convergence information (default=`false`).
- `cb`: A callback function for extending functionality.

See also: [`MVDA.anneal!`](@ref) for additional keyword arguments applied at the annealing step.
"""
function StatsBase.fit!(algorithm::AbstractMMAlg, problem::MVDAProblem, epsilon::Real, lambda::Real, s::Real,
    _extras_::T=nothing;
    maxrhov::Int=DEFAULT_MAXRHOV,
    dtol::Real=DEFAULT_DTOL,
    rtol::Real=DEFAULT_RTOL,
    rho_init::Real=DEFAULT_RHO_INIT,
    rho_max::Real=DEFAULT_RHO_MAX,
    rhof::Function=DEFAULT_ANNEALING,
    callback::F=DEFAULT_CALLBACK,
    kwargs...) where{T,F}
    # Check for missing data structures.
    extras = __mm_init__(algorithm, problem, _extras_)

    # Get problem info and extra data structures.
    @unpack coeff, coeff_prev, coeff_proj = problem
    @unpack projection = extras

    # Initialize ρ and iteration count.
    rho, iters = rho_init, 0

    # Fix hyperparameters.
    k = sparsity_to_k(problem, s)
    hyperparams = (;epsilon=epsilon, lambda=lambda, rho=rho, k=k)
    
    # Update data structures due to hyperparameters.
    __mm_update_rho__(algorithm, problem, extras, lambda, rho)

    # Check initial values for loss, objective, distance, and norm of gradient.
    apply_projection(projection, problem, k)
    state = evaluate_objective!(problem, extras, epsilon, lambda, rho)
    callback((0, state), problem, hyperparams)
    old = state.distance

    for iter in 1:maxrhov
        # Solve minimization problem for fixed rho.
        (inner_iters, state) = anneal!(algorithm, problem, epsilon, lambda, s, rho, extras;
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
        rho = ifelse(iter < maxrhov, rhof(rho, iter, rho_max), rho)
    end
    
    # Project solution to the constraint set.
    apply_projection(projection, problem, k)
    state = evaluate_objective!(problem, extras, epsilon, lambda, rho)

    return ((iters, state), rho)
end

"""
    anneal!(algorithm, problem, ϵ, ρ, s, [extras], [update_extras]; kwargs...)

Same as `anneal(algorithm, problem, ϵ, ρ, s)`, but with preallocated data structures in `extras`.

!!! Note
    The caller should specify whether to update data structures depending on `s` and `ρ` using `update_extras[1]` and `update_extras[2]`, respectively.

    Convergence is determined based on the rule `gradsq < gtol`, where `gradsq` is squared Euclidean norm of the gradient and `gtol` is a tolerance parameter.

!!! Tip
    The `extras` argument can be constructed using `extras = __mm_init__(algorithm, problem, nothing)`.

# Keyword Arguments

- `ninner`: The maximum number of iterations (default=`10^4`).
- `gtol`: An absoluate tolerance parameter on the squared Euclidean norm of the gradient (default=`1e-6`).
- `nesterov_threshold`: The number of early iterations before applying Nesterov acceleration (default=`10`).
- `verbose`: Print convergence information (default=`false`).
- `cb`: A callback function for extending functionality.
"""
function anneal!(algorithm::AbstractMMAlg, problem::MVDAProblem, epsilon::Real, lambda::Real, s::Real, rho::Real,
    _extras_::T=nothing;
    maxiter::Int=10^4,
    gtol::Real=1e-6,
    nesterov::Int=10,
    callback::F=DEFAULT_CALLBACK,
    ) where {T,F}
    # Check for missing data structures.
    extras = __mm_init__(algorithm, problem, _extras_)

    # Get problem info and extra data structures.
    @unpack coeff, coeff_prev, coeff_proj = problem
    @unpack projection = extras

    # Fix hyperparameters.
    k = sparsity_to_k(problem, s)
    hyperparams = (;epsilon=epsilon, lambda=lambda, rho=rho, k=k)

    # Update data structures due to hyperparameters.
    __mm_update_rho__(algorithm, problem, extras, lambda, rho)

    # Check initial values for loss, objective, distance, and norm of gradient.
    apply_projection(projection, problem, k)
    state = evaluate_objective!(problem, extras, epsilon, lambda, rho)
    old = state.objective

    if state.gradient < gtol
        return (0, state)
    end

    # Initialize iteration counts.
    iters = 0
    nesterov_iter = 1
    for iter in 1:maxiter
        iters += 1

        # Apply an algorithm map to update estimates.
        __mm_iterate__(algorithm, problem, extras, epsilon, lambda, rho, k)

        # Update loss, objective, distance, and gradient.
        apply_projection(projection, problem, k)
        state = evaluate_objective!(problem, extras, epsilon, lambda, rho)
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

    return (iters, state)
end
