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

# sparse model
function evaluate_gradient!(problem::MVDAProblem, lambda, rho)
    @unpack coeff, res, grad, intercept = problem
    n, _, _ = probdims(problem)
    A = design_matrix(problem)
    B = coeff.slope
    T = floattype(problem)

    alpha, beta, gamma = convert(T, 1/n), convert(T, lambda), convert(T, rho)

    if intercept
        mean!(grad.intercept, res.loss')
        grad.intercept .*= -one(T)
    end
    copyto!(grad.slope, res.dist)
    BLAS.gemm!('T', 'N', -alpha, A, res.loss, -gamma, grad.slope)
    BLAS.axpy!(beta, B, grad.slope)

    return nothing
end

# regularized model
function evaluate_gradient!(problem::MVDAProblem, lambda)
    @unpack coeff, res, grad, intercept = problem
    n, _, _ = probdims(problem)
    A = design_matrix(problem)
    B = coeff.slope
    T = floattype(problem)

    alpha, beta = convert(T, 1/n), convert(T, lambda)

    if intercept
        mean!(grad.intercept, res.loss')
        grad.intercept .*= -one(T)
    end
    copyto!(grad.slope, B)
    BLAS.gemm!('T', 'N', -alpha, A, res.loss, beta, grad.slope)

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
Evaluate the penalized least squares criterion. Also updates the gradient.
This assumes that projections have been handled externally.
"""
function evaluate_objective!(problem::MVDAProblem, extras, epsilon, lambda, rho)
    @unpack n, coeff, res, grad = problem

    evaluate_residuals!(problem, extras, epsilon, true, true)
    evaluate_gradient!(problem, lambda, rho)

    B, R, Q, G = coeff.slope, res.loss, res.dist, grad
    risk = 1//n * dot(R, R)
    penalty = dot(B, B)
    loss = 1//2 * (risk + lambda*penalty)
    distsq = dot(Q, Q)
    obj = loss + 1//2*rho*distsq
    gradsq = dot(G, G)

    return __eval_result__(risk, loss, obj, penalty, distsq, gradsq)
end

function evaluate_objective!(problem::MVDAProblem, extras, epsilon, lambda)
    @unpack n, coeff, res, grad = problem

    evaluate_residuals!(problem, extras, epsilon, true, false)
    evaluate_gradient!(problem, lambda)

    B, R, G = coeff.slope, res.loss, grad
    risk = 1//n * dot(R, R)
    penalty = dot(B, B)
    loss = 1//2 * (risk + lambda*penalty)
    distsq = zero(floattype(problem))
    obj = loss
    gradsq = dot(G, G)

    return __eval_result__(risk, loss, obj, penalty, distsq, gradsq)
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

"""
Apply a projection to model coefficients.
"""
function apply_projection(projection, problem, k)
    @unpack coeff, coeff_proj = problem
    copyto!(coeff_proj.slope, coeff.slope)
    copyto!(coeff_proj.intercept, coeff.intercept)
    projection(coeff_proj.slope, k)
    return coeff_proj
end

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

function make_sparsity_grid(n, nbins, len=round(Int, 1e1 ^ (log10(n)-1)))
    function sumr(r, K)
        if r != 1
            Float64(r/(r-1) * (r^K - 1))
        else
            Float64(K)
        end
    end

    xs = Float64[]
    push!(xs, 0.0)

    if n > 100
        c = fzero(c -> (1-c^(nbins+1))/(1-c) - 2, 0.0, 1-eps())
        r = fzero(r -> sumr(r, nbins) - len, 0.0, Float64(n))
        N = zeros(Int, nbins)
        for k in 1:nbins-1
            N[k] = max(1, floor(Int, r^k))
        end
        N[end] = max(sum(N)-len, len-sum(N))
        for (k, nk) in enumerate(N)
            a = xs[end]
            b = xs[end] + c^k
            vals = range(a, b, length=nk+1)
            foreach(Base.Fix1(push!, xs), vals[2:end])
        end
    else
        for i in 0:n-1
            push!(xs, i/n)
        end
    end
    ys = unique!(round.(Int, (1 .- xs) .* n))
    xs = sort!(1 .- ys ./ n)
    filter!(!isequal(1.0), xs)
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
