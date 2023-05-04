# Y = B'x - b0
function predicted_response!(Y, X, B, b0, intercept)
    T = eltype(Y)
    if intercept
        foreach(Base.Fix2(copyto!, b0), eachrow(Y))
        BLAS.gemm!('N', 'N', one(T), X, B, one(T), Y)
    else
        BLAS.gemm!('N', 'N', one(T), X, B, zero(T), Y)
    end
    return nothing
end
#
# r = y - yhat
# z = yhat if |r|_2 <= epsilon
# z = w*y + (1-w)*yhat if |r|_2 > epsilon
#
# R enters as Yhat
#
function shifted_response!(Z, Y, R, epsilon)
    num_julia_threads = Threads.nthreads()

    if num_julia_threads == 1
        for i in axes(Y, 1)
            z, y, r = view(Z, i, :), view(Y, i, :), view(R, i, :)
            shifted_response!(z, y, r, epsilon)
        end
    else
        num_BLAS_threads = BLAS.get_num_threads()
        try
            BLAS.set_num_threads(1)
            @batch per=core for i in axes(Y, 1)
                z, y, r = view(Z, i, :), view(Y, i, :), view(R, i, :)
                shifted_response!(z, y, r, epsilon)
            end
        finally
            BLAS.set_num_threads(num_BLAS_threads)
        end
    end

    return nothing
end

function shifted_response!(z::AbstractVector, y::AbstractVector, r::AbstractVector, epsilon)
    T = eltype(z)

    # Compute residual R = Y - A*B - 1b₀ᵀ
    copyto!(z, r)
    BLAS.axpby!(one(T), y, -one(T), r)

    # Compute norm of residual and set the weight for projection onto deadzone.
    normr = norm(r)
    w = ifelse(normr > epsilon, (normr-epsilon)/normr, zero(T))

    # Set Z = W*Y + (1-W)∘Ŷ
    BLAS.axpby!(w, y, 1-w, z)

    # Set R = W*R
    BLAS.scal!(w, r)
end

"""
Generic template for evaluating residuals.
This assumes that projections have been handled externally.
The following flags control how residuals are evaluated:

+ `need_main`: If `true`, evaluates regression residuals.
+ `need_dist`: If `true`, evaluates difference between parameters and their projection.

**Note**: The values for each flag should be known at compile-time!
"""
function evaluate_residuals!(problem::MVDAProblem, extras, epsilon, need_loss::Bool, need_dist::Bool)
    @unpack Y, coeff, coeff_proj, res, intercept = problem
    @unpack Z = extras
    T = floattype(problem)
    A = design_matrix(problem)
    B, b0 = coeff.slope, coeff.intercept
    R, Q = res.loss, res.dist

    if need_loss
        predicted_response!(R, A, B, b0, intercept) # R = A*B + 1*b₀ᵀ = Ŷ
        shifted_response!(Z, Y, R, epsilon)         # Z = W∘Y + (1-W)∘Ŷ, R = W∘(Y - Ŷ)
    end

    if need_dist
        # distance residuals, P(B) - B
        copyto!(Q, coeff_proj.slope)
        BLAS.axpy!(-one(T), B, Q)
    end

    return nothing
end

function __eval_result__(risk, loss, obj, penalty, distsq, gradsq)
    return (;
        risk=risk,
        loss=loss,
        objective=obj,
        penalty=penalty,
        distance=sqrt(distsq),
        gradient=sqrt(gradsq)
    )
end

"""
Apply acceleration to the current iterate `x` based on the previous iterate `y`
according to Nesterov's method with parameter `r=3` (default).
"""
function nesterov_acceleration!(x::T, y::T, iter::Integer, needs_reset::Bool, r::Int=3) where T <: AbstractArray
    if needs_reset # Reset acceleration scheme
        copyto!(y, x)
        iter = 1
    else # Nesterov acceleration 
        c = (iter - 1) / (iter + r - 1)
        for i in eachindex(x)
            xi, yi = x[i], y[i]
            zi = xi + c * (xi - yi)
            x[i], y[i] = zi, xi
        end
        iter += 1
    end

    return iter
end

function nesterov_acceleration!(x::T, y::T, args...) where T <: Coefficients
    nesterov_acceleration!(x.slope, y.slope, args...)
    nesterov_acceleration!(x.intercept, y.intercept, args...)
end

"""
Map a sparsity level `s` to an integer `k`, assuming `n` elements.
"""
sparsity_to_k(problem::MVDAProblem, s) = sparsity_to_k(problem.kernel, problem, s)
sparsity_to_k(::Nothing, problem::MVDAProblem, s) = round(Int, (1-s) * problem.p)
sparsity_to_k(::Kernel, problem::MVDAProblem, s) = round(Int, (1-s) * problem.n)

struct GeometricProression <: Function
    multiplier::Float64
end

function (f::GeometricProression)(rho, iter, rho_max)
    convert(typeof(rho), min(rho_max, rho * f.multiplier))
end

"""
Define a geometric progression recursively by the rule
```
    rho_new = min(rho_max, rho * multiplier)
```
The result is guaranteed to have type `typeof(rho)`.
"""
function geometric_progression(multiplier::Real=1.2)
    return GeometricProression(multiplier)
end

function make_sparsity_grid(n, len)
    xs = Float64[]
    push!(xs, 0.0)
    if n > 100
        r = exp(log(n) / len)
        for k in 1:len-1
            push!(xs, 1 - r^k / n)
        end
    else
        for i in 0:n
            push!(xs, i/n)
        end
    end
    ys = unique!(round.(Int, (1 .- xs) .* n))
    filter!(<=(n), ys)
    xs = sort!(1 .- ys ./ n)
    return xs
end

function make_regular_log10_grid(a, b, m)
    xs = Float64[]
    
    for c in a:b-1
        r = range(c, c+1, length=m)
        for ri in r
            push!(xs, 10.0 ^ ri)
        end
    end

    return unique!(xs)
end

function make_log10_grid(a, b, n)
    if a == -Inf
        [0.0; 10.0 .^ range(log10(eps()), b, length=n-1)]
    else
        10.0 .^ range(a, b, length=n)
    end
end
