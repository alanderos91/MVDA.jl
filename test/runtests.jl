using MVDA
using LinearAlgebra, Random, StableRNGs, Parameters, Statistics
using Test

df = MVDA.dataset("iris")
target, X = Vector(df.target), Matrix(df[!,2:end])
n, p, c = length(target), size(X, 2), length(unique(target))
prob = MVDAProblem(target, X)

@testset "MVDAProblem" begin
    # coefficient shape
    for field in (:coeff, :coeff_prev, :proj, :grad)
        arr = getfield(prob, field)
        @test size(arr.all, 1) == p && size(arr.all, 2) == c-1
        @test all(j -> length(arr.dim[j]) == p, eachindex(arr.dim))
    end

    # residual shape
    r1 = prob.res.main.all
    r2 = prob.res.dist.all
    r3 = prob.res.weighted.all
    @test size(r1, 1) == n && size(r1, 2) == c-1
    @test size(r2, 1) == p && size(r2, 2) == c-1
    @test size(r3, 1) == n && size(r3, 2) == c-1

    # verify problem dimensions
    dims = MVDA.probdims(prob)
    @test n == dims[1] && p == dims[2] && c == dims[3]

    # check vertex definitions
    v = prob.vertex
    d = [norm(v[j] - v[i]) for i in 1:c for j in i+1:c]
    dmin, dmax = extrema(d)
    @test all(j -> norm(v[j]) ≈ 1, eachindex(v))    # vertices lie on unit circle
    @test dmin ≈ dmax                               # vertices are equidistant
end

@testset "Projections" begin
    @testset "L0" begin
        for _ in 1:10
            x = 10 * randn(1000)
            k = 50
            P = MVDA.ApplyL0Projection(collect(axes(x, 1)))
            x_proj = P(x, k)

            perm = sortperm(x, lt=isless, by=abs, rev=true, order=Base.Order.Forward)
            @test sort!(P.idx[1:k]) == sort!(perm[1:k])
        end
    end
end

@testset "Evaluation" begin
    @unpack Y, X, coeff, res, proj, grad = prob
    B = coeff.all
    P = proj.all
    T = Float64

    # Initialize coefficients and parameters; do we need a stable RNG?
    rng = StableRNG(1903)
    randn!(rng, B)
    ϵ = 1e-3
    ρ = 4.7
    k = [p for _ in 1:c-1]

    # Set projection.
    apply_projection = MVDA.ApplyL0Projection(collect(1:p))
    copyto!(P, B)
    for j in 1:c-1
        apply_projection(proj.dim[j], k[j])
    end

    # Compute residuals and gradient
    a = 1 / sqrt(n)
    b = [1 / sqrt( (c-1)*(p-k[j]+1) ) for j in eachindex(k)]
    D = Diagonal(b)
    A = [ [a*X; b[j]*I] for j in eachindex(b) ]

    R1 = Y - X*B
    R2 = (P - B)
    R = [R1; R2]
    w = zeros(n)
    Z = zeros(n, c-1)
    XB = X*B

    for i in 1:n
        yᵢ = @view Y[i,:]
        rᵢ = @view R1[i,:]
        zᵢ = @view Z[i,:]
        XBᵢ = @view XB[i,:]

        normrᵢ = norm(rᵢ)
        wᵢ = ifelse(normrᵢ ≤ ϵ, zero(T), (normrᵢ-ϵ)/normrᵢ)
        zᵢ .= ifelse(normrᵢ ≤ ϵ, XBᵢ, wᵢ*yᵢ + (1-wᵢ)*XBᵢ)
        w[i] = wᵢ
    end

    R1_scaled = 1/sqrt(n) * Diagonal(w) * R1
    R2_scaled = R2 * D
    R_scaled = [R1_scaled; R2_scaled]
    Rtmp = [R1_scaled; ρ*R2_scaled]

    G = similar(B)
    for j in eachindex(A)
        G[:,j] .= -A[j]' * Rtmp[:,j]
    end

    extras = MVDA.__mm_init__(MMSVD(), prob, nothing)
    MVDA.__mm_update_sparsity__(MMSVD(), prob, ϵ, ρ, k, extras)
    MVDA.__mm_update_rho__(MMSVD(), prob, ϵ, ρ, k, extras)

    @testset "Residuals" begin
        MVDA.__evaluate_residuals__(prob, ϵ, extras, true, true, true)
        @test all(j -> res.main.dim[j] ≈ 1/sqrt(n) * view(R1, :, j), 1:c-1)
        @test all(j -> res.dist.dim[j] ≈ view(R2_scaled, :, j), 1:c-1)
        @test all(j -> res.weighted.dim[j] ≈ view(R1_scaled, :, j), 1:c-1)
        @test all(i -> view(extras.Z, i, :) ≈ view(Z, i, :), 1:n)
    end

    @testset "Gradient" begin
        MVDA.__evaluate_gradient__(prob, ρ, extras)
        @test all(j -> grad.dim[j] ≈ view(G, :, j), 1:c-1)
    end

    @testset "Objective" begin
        loss, obj, dist, gradsq = MVDA.__evaluate_objective__(prob, ϵ, ρ, extras)
        
        @test loss ≈ norm(R1_scaled)^2
        @test obj ≈ 1//2 * (norm(R1_scaled)^2 + ρ * norm(R2_scaled)^2)
        @test dist ≈ norm(R2_scaled)^2
        @test gradsq ≈ norm(G)^2
    end
end

# @testset "Descent Property" begin
#     problem_size = (
#         (500, 600, 50, 10,), # underdetermined, n < p
#         (600, 500, 50, 10,), # overdetermined, n > p
#         (500, 50, 600, 10,),
#     )
    
#     for (n,p,d,k) in problem_size
#         X, Z = randn(n, p), randn(p, d)
#         X .= (X .- mean(X, dims=1)) ./ std(X, dims=1)
#         Z .= (Z .- mean(Z, dims=1)) ./ std(Z, dims=1)
    
#         α0 = zeros(d)
#         α0[1:k] = rand((-1,1), k) * 10 + randn(k)
#         β0 = Z*α0 + randn(p)
#         γ0 = β0 - Z*α0
#         y = X*β0
    
#         problem = ProxDistProblem(y, X, Z)
#         λ = 1e1
#         ρ = 1.234
#         s = 0.0
    
#         @testset "$(algorithm)" for algorithm in (SD(), MMSVD(), MMBCD(), MMCD(),)
#             function test_descent_property(threshold)
#                 copyto!(problem.coeff.int, γ0)
#                 copyto!(problem.coeff.ext, α0)
    
#                 _, _, _, obj0A, _, _ = hreg(algorithm, problem, λ, ρ, s, ninner=0, gtol=1e-6, nesterov_threshold=threshold)
#                 _, _, _, obj0B, _, _ = hreg(algorithm, problem, λ, ρ, s, ninner=0, gtol=1e-6, nesterov_threshold=threshold)
#                 _, _, _, obj1, _, _ = hreg(algorithm, problem, λ, ρ, s, ninner=1, gtol=1e-6, nesterov_threshold=threshold)
#                 _, _, _, obj2, _, _ = hreg(algorithm, problem, λ, ρ, s, ninner=1, gtol=1e-6, nesterov_threshold=threshold)
#                 _, _, _, obj100, _, _ = hreg(algorithm, problem, λ, ρ, s, ninner=98, gtol=1e-6, nesterov_threshold=threshold)
    
#                 @test obj0A == obj0B # no iterations
#                 @test obj0A > obj1   # decrease after 1 iteration
#                 @test obj1 > obj2    # decrease after 1 iteration
#                 @test obj0A > obj100 # decrease after 100 iterations

#                 @test all(!isnan, problem.coeff.all)
#             end
#             # w/o Nesterov acceleration
#             test_descent_property(100)
    
#             # w/ Nesterov acceleration
#             test_descent_property(10)
#         end
#     end
# end
