"""
Generic template for evaluating residuals.
This assumes that projections have been handled externally.
The following flags control how residuals are evaluated:

+ `need_main`: If `true`, evaluates regression residuals.
+ `need_dist`: If `true`, evaluates difference between parameters and their projection.

**Note**: The values for each flag should be known at compile-time!
"""
function __evaluate_residuals__(problem, ϵ, extras, need_main::Bool, need_dist::Bool, need_z::Bool)
    @unpack Y, X, coeff, proj, res = problem
    @unpack Z = extras
    T = floattype(problem)

    if need_main
        # main residuals, √(1/n) * (Y - X*B)
        a = 1 / sqrt(size(Y, 1))
        mul!(res.main.all, X, coeff.all)
        need_z && copyto!(Z, res.main.all)
        axpby!(a, Y, -a, res.main.all)

        # weighted residuals, W^1/2 * (Y - X*B)
        for i in axes(Y, 1)
            yᵢ = view(Y, i, :)
            rᵢ = res.main.sample[i]
            wrᵢ = res.weighted.sample[i]
            normrᵢ = norm(rᵢ) / a
            wᵢ = ifelse(normrᵢ ≤ ϵ, zero(T), (normrᵢ-ϵ)/normrᵢ)
            @. wrᵢ = wᵢ * rᵢ
            if need_z
                # zᵢ = X*B if norm(rᵢ) ≤ ϵ
                # zᵢ = wᵢ*yᵢ + (1-wᵢ)*X*B otherwise
                zᵢ = view(Z, i, :)
                axpby!(wᵢ, yᵢ, 1-wᵢ, zᵢ)
            end
        end
    end

    if need_dist
        # res_dist = P(B) - B
        copyto!(res.dist.all, proj.all)
        axpby!(-one(T), coeff.all, one(T), res.dist.all)
    end

    return nothing
end

"""
Evaluate the gradiant of the regression problem. Assumes residuals have been evaluated.
"""
function __evaluate_gradient__(problem, ρ, extras)
    @unpack X, res, grad = problem

    for j in eachindex(grad.dim)
        # ∇g_ρ(B ∣ Bₘ)ⱼ = -[aXᵀ bⱼI] * Rₘ,ⱼ
        a = 1 / sqrt(size(X, 1))
        b = ρ
        mul!(grad.dim[j], X', res.weighted.dim[j])
        axpby!(-b, res.dist.dim[j], -a, grad.dim[j])
    end

    return nothing
end

function __evaluate_gradient_reg__(problem, λ, extras)
    @unpack X, res, grad = problem

    for j in eachindex(grad.dim)
        # ∇g_ρ(B ∣ Bₘ)ⱼ = -[aXᵀ λI] * Rₘ,ⱼ
        a = 1 / sqrt(size(X, 1))
        mul!(grad.dim[j], X', res.weighted.dim[j])
        axpby!(λ, problem.coeff.dim[j], -a, grad.dim[j])
    end

    return nothing
end

"""
Evaluate the penalized least squares criterion. Also updates the gradient.
This assumes that projections have been handled externally.
"""
function __evaluate_objective__(problem, ϵ, ρ, extras)
    @unpack res, grad = problem

    __evaluate_residuals__(problem, ϵ, extras, true, true, false)
    __evaluate_gradient__(problem, ρ, extras)

    loss = norm(res.weighted.all)^2 # 1/n * ∑ᵢ (Zᵢ - Bᵀxᵢ)²
    dist = norm(res.dist.all)^2     # ∑ⱼ (P(B)ⱼ - Bⱼ)²
    obj = 1//2 * (loss + ρ * dist)
    gradsq = norm(grad.all)^2

    return IterationResult(loss, obj, dist, gradsq)
end

function __evaluate_objective_reg__(problem, ϵ, λ, extras)
    @unpack res, grad = problem

    __evaluate_residuals__(problem, ϵ, extras, true, false, false)
    __evaluate_gradient_reg__(problem, λ, extras)

    loss = norm(res.weighted.all)^2 # 1/n * ∑ᵢ (Zᵢ - Bᵀxᵢ)²
    objective = 1//2 * (loss + λ * norm(problem.coeff.all))
    gradsq = norm(grad.all)^2

    return IterationResult(loss, objective, 0.0, gradsq)
end

"""
Apply acceleration to the current iterate `x` based on the previous iterate `y`
according to Nesterov's method with parameter `r=3` (default).
"""
function __apply_nesterov__!(x, y, iter::Integer, needs_reset::Bool, r::Int=3)
    if needs_reset # Reset acceleration scheme
        copyto!(y, x)
        iter = 1
    else # Nesterov acceleration 
        γ = (iter - 1) / (iter + r - 1)
        @inbounds for i in eachindex(x)
            xi, yi = x[i], y[i]
            zi = xi + γ * (xi - yi)
            x[i], y[i] = zi, xi
        end
        iter += 1
    end

    return iter
end

"""
Map a sparsity level `s` to an integer `k`, assuming `n` elements.
"""
sparsity_to_k(s, n) = round(Int, n * (1-s))

"""
Define a geometric progression recursively by the rule
```
    rho_new = min(rho_max, rho * multiplier)
```
The result is guaranteed to have type `typeof(rho)`.
"""
function geometric_progression(rho, iter, rho_max, multiplier::Real=1.2)
    return convert(typeof(rho), min(rho_max, rho * multiplier))
end

"""
Default error message for missing methods. For internal use only.
"""
not_implemented(alg, msg) = error(string(msg, " not implemented for ", alg, "."))

"""
Placeholder for callbacks in main functions.
"""
__do_nothing_callback__(iter, problem, lambda, rho, k, history) = nothing
__do_nothing_callback__(fold, problem, train_problem, data, lambda, sparsity, model_size, result) = nothing

__svd_wrapper__(A::StridedMatrix) = svd(A)
__svd_wrapper__(A::AbstractMatrix) = svd!(copy(A))

function prediction_error(model, train_set, validation_set, test_set)
    # Extract data for each set.
    Tr_Y, Tr_X = train_set
    V_Y, V_X = validation_set
    T_Y, T_X = test_set

    Tr_label = map(yᵢ -> model.vertex2label[yᵢ], eachrow(Tr_Y))
    V_label = map(yᵢ -> model.vertex2label[yᵢ], eachrow(V_Y))
    T_label = map(yᵢ -> model.vertex2label[yᵢ], eachrow(T_Y))

    # Make predictions on each subset.
    Tr_call = classify(model, Tr_X)
    V_call = classify(model, V_X)
    T_call = classify(model, T_X)

    # Evaluate errors on each subset.
    Tr = 100 * (1 - sum(Tr_call .== Tr_label) / length(Tr_label))
    V = 100 * (1 - sum(V_call .== V_label) / length(V_label))
    T = 100 * (1 - sum(T_call .== T_label) / length(T_label))

    return [Tr, V, T]
end

struct IterationResult
    loss::Float64
    objective::Float64
    distance::Float64
    gradient::Float64
end

# destructuring
Base.iterate(r::IterationResult) = (r.loss, Val(:objective))
Base.iterate(r::IterationResult, ::Val{:objective}) = (r.objective, Val(:distance))
Base.iterate(r::IterationResult, ::Val{:distance}) = (r.distance, Val(:gradient))
Base.iterate(r::IterationResult, ::Val{:gradient}) = (r.gradient, Val(:done))
Base.iterate(r::IterationResult, ::Val{:done}) = nothing

struct SubproblemResult
    iters::Int
    loss::Float64
    objective::Float64
    distance::Float64
    gradient::Float64
end

function SubproblemResult(iters, r::IterationResult)
    return SubproblemResult(iters, r.loss, r.objective, r.distance, r.gradient)
end

# destructuring
Base.iterate(r::SubproblemResult) = (r.iters, Val(:loss))
Base.iterate(r::SubproblemResult, ::Val{:loss}) = (r.loss, Val(:objective))
Base.iterate(r::SubproblemResult, ::Val{:objective}) = (r.objective, Val(:distance))
Base.iterate(r::SubproblemResult, ::Val{:distance}) = (r.distance, Val(:gradient))
Base.iterate(r::SubproblemResult, ::Val{:gradient}) = (r.gradient, Val(:done))
Base.iterate(r::SubproblemResult, ::Val{:done}) = nothing
