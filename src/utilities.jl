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
        @inbounds @simd for i in eachindex(x)
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
Placeholder for callbacks in main functions.
"""
__do_nothing_callback__(iter, problem, lambda, rho, k, history) = nothing
__do_nothing_callback__(fold, problem, train_problem, data, lambda, sparsity, model_size, result) = nothing
