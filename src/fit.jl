function solve!(f::AbstractVDAModel, args...; kwargs...)
    solve_unconstrained!(f, args...; kwargs...)
end

function solve!(f::PenalizedObjective{LOSS,SqDistPenalty}, args...; kwargs...) where LOSS
    solve_constrained!(f, args...; kwargs...)
end

function solve!(f::AbstractVDAModel, algorithm::PGD, args...; kwargs...)
    solve_constrained!(f, algorithm, args...; kwargs...)
end

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
    stuck_iters = 0
    old_grad = state.gradient

    for iter in 1:maxiter
        iters += 1

        # Apply an algorithm map to update estimates.
        __mm_iterate__(algorithm, f, problem, extras, hyperparams)

        # Update loss, objective, and gradient.
        state = evaluate_model!(f, problem, extras, hyperparams)
        callback((iter, state), problem, hyperparams)

        # Assess convergence.
        obj = state.objective
        if state.gradient < gtol || stuck_iters > 2
            break
        elseif iter < maxiter
            if isapprox(old_grad, state.gradient; rtol=1e-6)
                stuck_iters += 1
                needs_reset = true
            else
                stuck_iters = 0
                needs_reset = iter < nesterov || obj > old
            end
            nesterov_iter = nesterov_acceleration!(coeff, coeff_prev, nesterov_iter, needs_reset)
            old = obj
            old_grad = state.gradient
        end
    end
    copyto!(coeff_proj.slope, coeff.slope)
    copyto!(coeff_proj.intercept, coeff.intercept)

    return ((iters, state), zero(floatT))
end

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
    rng::AbstractRNG=Random.default_rng(),
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
        if rho >= 1e2 && (dist < dtol || abs(dist - old) < rtol * (1 + old))
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

function solve_constrained!(f::AbstractVDAModel, algorithm::PGD, problem::MVDAProblem, hparams,
    _extras_::T=nothing;
    projection_type::Type=HomogeneousL0Projection,
    maxiter::Int=DEFAULT_MAXITER,
    gtol::Real=DEFAULT_GTOL,
    nesterov::Int=DEFAULT_NESTEROV,
    callback::F=DEFAULT_CALLBACK,
    rng::AbstractRNG=Random.default_rng(),
    kwargs...,
    ) where {T,F}
    # Check for missing data structures.
    extras = __mm_init__(algorithm, (projection_type, rng), problem, _extras_)
    floatT = floattype(problem)

    # Get problem info and extra data structures.
    @unpack coeff, coeff_prev, coeff_proj = problem
    hyperparams = (; hparams..., rho=0.0)

    # Update data structures due to hyperparameters.
    __mm_update_datastructures__(algorithm, f, problem, extras, hyperparams)

    # Check initial values for loss, objective, distance, and norm of gradient.
    rand!(rng, coeff_prev.slope)
    rand!(rng, coeff_prev.intercept)
    state = evaluate_model!(f, problem, extras, hyperparams)
    gradient_mapping = evaluate_gradient_mapping!(problem, 1.0)
    state = (; state..., gradient=gradient_mapping,) # gradient is not informative here
    callback((0, state), problem, hyperparams)
    old = state.objective

    # Initialize iteration counts.
    iters = 0
    nesterov_iter = 1

    for iter in 1:maxiter
        iters += 1

        # Apply an algorithm map to update estimates.
        t = __mm_iterate__(algorithm, f, problem, extras, hyperparams)

        # Update loss, objective, and gradient.
        state = evaluate_model!(f, problem, extras, hyperparams)
        gradient_mapping = evaluate_gradient_mapping!(problem, t)
        state = (; state..., gradient=gradient_mapping,) # gradient is not informative here
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

function solve_unconstrained!(f::PenalizedObjective{LOSS,SqDistPenalty}, algorithm::AbstractMMAlg, problem::MVDAProblem, hyperparams,
    _extras_::T=nothing;
    projection_type::Type=HomogeneousL0Projection,
    maxiter::Int=10^4,
    gtol::Real=1e-6,
    nesterov::Int=10,
    callback::F=DEFAULT_CALLBACK,
    rng::AbstractRNG=Random.default_rng(),
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
