"""
    fit(algorithm, problem, ϵ, s; kwargs...)

Solve optimization problem at sparsity level `s` using a deadzone of size `ϵ`.

The solution is obtained via a proximal distance `algorithm` that gradually anneals parameter estimates
toward the target sparsity set.
"""
function fit(algorithm::AbstractMMAlg, problem::MVDAProblem, ϵ::Real, s::Real; kwargs...)
    extras = __mm_init__(algorithm, problem, nothing) # initialize extra data structures
    MVDA.fit!(algorithm, problem, ϵ, s, extras, (true,false,); kwargs...)
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
function fit!(algorithm::AbstractMMAlg, problem::MVDAProblem, ϵ::Real, s::Real,
    extras=nothing,
    update_extras::NTuple{2,Bool}=(true,false,);
    nouter::Int=100,
    dtol::Real=1e-6,
    rtol::Real=1e-6,
    rho_init::Real=1.0,
    rho_max::Real=1e8,
    rhof::Function=DEFAULT_ANNEALING,
    verbose::Bool=false,
    cb::Function=DEFAULT_CALLBACK,
    kwargs...)
    # Check for missing data structures.
    if extras isa Nothing
        error("Detected missing data structures for algorithm $(algorithm).")
    end

    # Get problem info and extra data structures.
    @unpack intercept, coeff, coeff_prev, proj = problem
    @unpack projection = extras
    
    # Fix model size(s).
    k = sparsity_to_k(problem, s)

    # Initialize ρ and iteration count.
    ρ, iters = rho_init, 0

    # Update data structures due to hyperparameters.
    update_extras[1] && __mm_update_sparsity__(algorithm, problem, ϵ, ρ, k, extras)
    update_extras[2] && __mm_update_rho__(algorithm, problem, ϵ, ρ, k, extras)

    # Check initial values for loss, objective, distance, and norm of gradient.
    apply_projection(projection, problem, k)
    init_result = __evaluate_objective__(problem, ϵ, ρ, extras)
    result = SubproblemResult(0, init_result)
    cb(0, problem, ϵ, ρ, k, result)
    old = sqrt(result.distance)

    for iter in 1:nouter
        # Solve minimization problem for fixed rho.
        verbose && print("\n",iter,"  ρ = ",ρ)
        result = MVDA.anneal!(algorithm, problem, ϵ, ρ, s, extras, (false,true,); verbose=verbose, cb=cb, kwargs...)

        # Update total iteration count.
        iters += result.iters

        cb(iter, problem, ϵ, ρ, k, result)

        # Check for convergence to constrained solution.
        dist = sqrt(result.distance)
        if dist < dtol || abs(dist - old) < rtol * (1 + old)
            break
        else
          old = dist
        end
                
        # Update according to annealing schedule.
        ρ = ifelse(iter < nouter, rhof(ρ, iter, rho_max), ρ)
    end
    
    # Project solution to the constraint set.
    apply_projection(projection, problem, k)
    loss, obj, dist, gradsq = __evaluate_objective__(problem, ϵ, ρ, extras)

    if verbose
        print("\n\niters = ", iters)
        print("\n∑ᵢ max{0, |yᵢ-Bᵀxᵢ|₂-ϵ}² = ", loss)
        print("\nobjective  = ", obj)
        print("\ndistance   = ", sqrt(dist))
        println("\n|gradient| = ", sqrt(gradsq))
    end

    return SubproblemResult(iters, loss, obj, dist, gradsq)
end

"""
    anneal(algorithm, problem, ϵ, ρ, s; kwargs...)

Solve the `ρ`-penalized optimization problem at sparsity level `s` with deadzone `ϵ`.
"""
function anneal(algorithm::AbstractMMAlg, problem::MVDAProblem, ϵ::Real, ρ::Real, s::Real; kwargs...)
    extras = __mm_init__(algorithm, problem, nothing)
    MVDA.anneal!(algorithm, problem, ϵ, ρ, s, extras, (true,true,); kwargs...)
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
function anneal!(algorithm::AbstractMMAlg, problem::MVDAProblem, ϵ::Real, ρ::Real, s::Real,
    extras=nothing,
    update_extras::NTuple{2,Bool}=(true,true);
    ninner::Int=10^4,
    gtol::Real=1e-6,
    nesterov_threshold::Int=10,
    verbose::Bool=false,
    cb::Function=DEFAULT_CALLBACK,
    kwargs...
    )
    # Check for missing data structures.
    if extras isa Nothing
        error("Detected missing data structures for algorithm $(algorithm).")
    end

    # Get problem info and extra data structures.
    @unpack intercept, coeff, coeff_prev, proj = problem
    @unpack projection = extras

    # Fix model size(s).
    k = sparsity_to_k(problem, s)

    # Update data structures due to hyperparameters.
    update_extras[1] && __mm_update_sparsity__(algorithm, problem, ϵ, ρ, k, extras)
    update_extras[2] && __mm_update_rho__(algorithm, problem, ϵ, ρ, k, extras)

    # Check initial values for loss, objective, distance, and norm of gradient.
    apply_projection(projection, problem, k)
    result = __evaluate_objective__(problem, ϵ, ρ, extras)
    cb(0, problem, ϵ, ρ, k, result)
    old = result.objective

    if sqrt(result.gradient) < gtol
        return SubproblemResult(0, result)
    end

    # Use previous estimates in case of warm start.
    copyto!(coeff.all, coeff_prev.all)

    # Initialize iteration counts.
    iters = 0
    nesterov_iter = 1
    verbose && @printf("\n%-5s\t%-8s\t%-8s\t%-8s\t%-8s", "iter.", "loss", "objective", "distance", "|gradient|")
    for iter in 1:ninner
        iters += 1

        # Apply the algorithm map to minimize the quadratic surrogate.
        __mm_iterate__(algorithm, problem, ϵ, ρ, k, extras)

        # Update loss, objective, distance, and gradient.
        apply_projection(projection, problem, k)
        result = __evaluate_objective__(problem, ϵ, ρ, extras)

        cb(iter, problem, ϵ, ρ, k, result)

        if verbose
            @printf("\n%4d\t%4.3e\t%4.3e\t%4.3e\t%4.3e", iter, result.loss, result.objective, sqrt(result.distance), sqrt(result.gradient))
        end

        # Assess convergence.
        obj = result.objective
        gradsq = sqrt(result.gradient)
        if gradsq < gtol
            break
        elseif iter < ninner
            needs_reset = iter < nesterov_threshold || obj > old
            nesterov_iter = __apply_nesterov__!(coeff.all, coeff_prev.all, nesterov_iter, needs_reset)
            old = obj
        end
    end
    # Save parameter estimates in case of warm start.
    copyto!(coeff_prev.all, coeff.all)

    return SubproblemResult(iters, result)
end

"""
```init!(algorithm, problem, ϵ, λ, [_extras_]; [maxiter=10^3], [gtol=1e-6], [nesterov_threshold=10], [verbose=false])```

Initialize a `problem` with its `λ`-regularized solution.
"""
function init!(algorithm::AbstractMMAlg, problem::MVDAProblem, ϵ, λ, _extras_=nothing;
    maxiter::Int=10^3,
    gtol::Real=1e-6,
    nesterov_threshold::Int=10,
    verbose::Bool=false,
    )
    # Check for missing data structures.
    extras = __mm_init__(algorithm, problem, _extras_)

    # Get problem info and extra data structures.
    @unpack coeff, coeff_prev, proj = problem

    # Update data structures due to hyperparameters.
    __mm_update_lambda__(algorithm, problem, ϵ, λ, extras)

    # Initialize coefficients.
    randn!(coeff.all)
    copyto!(coeff_prev.all, coeff.all)

    # Check initial values for loss, objective, distance, and norm of gradient.
    result = __evaluate_reg_objective__(problem, ϵ, λ, extras)
    old = result.objective

    if sqrt(result.gradient) < gtol
        return SubproblemResult(0, result)
    end

    # Initialize iteration counts.
    iters = 0
    nesterov_iter = 1
    verbose && @printf("\n%-5s\t%-8s\t%-8s\t%-8s", "iter.", "loss", "objective", "|gradient|")
    for iter in 1:maxiter
        iters += 1

        # Apply the algorithm map to minimize the quadratic surrogate.
        __reg_iterate__(algorithm, problem, ϵ, λ, extras)

        # Update loss, objective, and gradient.
        result = __evaluate_reg_objective__(problem, ϵ, λ, extras)

        if verbose
            @printf("\n%4d\t%4.3e\t%4.3e\t%4.3e", iter, result.loss, result.objective, sqrt(result.gradient))
        end

        # Assess convergence.
        obj = result.objective
        gradsq = sqrt(result.gradient)
        if gradsq < gtol
            break
        elseif iter < maxiter
            needs_reset = iter < nesterov_threshold || obj > old
            nesterov_iter = __apply_nesterov__!(coeff.all, coeff_prev.all, nesterov_iter, needs_reset)
            old = obj
        end
    end
    # Save parameter estimates in case of warm start.
    copyto!(coeff_prev.all, coeff.all)
    copyto!(proj.all, coeff.all)

    return SubproblemResult(iters, result)
end
