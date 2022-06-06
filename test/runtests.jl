using MVDA
using LinearAlgebra, Random, StableRNGs, Parameters, Statistics
using Test

@testset "Projections" begin
    @testset "L0" begin
        for _ in 1:10
            number_components = 1000
            x = 10 * randn(number_components)
            k = 50
            P = MVDA.L0Projection(number_components)
            xproj = P(copy(x), k)

            nzidx = findall(xi -> xi != 0, xproj)
            @test length(nzidx) ≤ k
            @test sort!(xproj[nzidx], by=abs, rev=true) == sort(x, by=abs, rev=true)[1:k]
        end
    end
end

function test_on_dataset(prob, target, X, k)
    n, p, c = length(target), size(X, 2), length(unique(target))
    has_intercept = prob.intercept

    @testset "MVDAProblem" begin
        # coefficient shape
        for field in (:coeff, :coeff_prev, :proj, :grad)
            arr = getfield(prob, field)
            @test size(arr.all, 1) == p + has_intercept && size(arr.all, 2) == c-1
            @test all(j -> length(arr.dim[j]) == p + has_intercept, eachindex(arr.dim))
        end
    
        # residual shape
        r1 = prob.res.main.all
        r2 = prob.res.dist.all
        r3 = prob.res.weighted.all
        @test size(r1, 1) == n && size(r1, 2) == c-1
        @test size(r2, 1) == p + has_intercept && size(r2, 2) == c-1
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
    
    @testset "Evaluation" begin
        @unpack Y, X, coeff, res, proj, grad = prob
        B = coeff.all
        P = proj.all
        T = Float64
    
        # Initialize coefficients and parameters
        rng = StableRNG(1903)
        randn!(rng, B)
        ϵ = 0.5 * sqrt(2*c/(c-1))
        ρ = 4.7
    
        # Initialize other data structures.
        extras = MVDA.__mm_init__(MMSVD(), prob, nothing)
        MVDA.__mm_update_sparsity__(MMSVD(), prob, ϵ, ρ, k, extras)
        MVDA.__mm_update_rho__(MMSVD(), prob, ϵ, ρ, k, extras)
        operator = extras.projection

        # Set projection.
        MVDA.apply_projection(operator, prob, k)

        # Compute residuals and gradient.
        XB = X*B
        R1 = Y - XB
        R2 = P - B

        normr = [norm(ri) for ri in eachrow(R1)]
        W = Diagonal( [ifelse(normr[i] ≤ ϵ, 0.0, (normr[i]-ϵ) / normr[i]) for i in eachindex(normr)] )
        R1_scaled = 1/sqrt(n) * W * R1
        Z = similar(Y)
        @views for i in axes(Z, 1)
            Z[i, :] .= ifelse(normr[i] ≤ ϵ, XB[i, :], W[i,i] * Y[i, :] + (1-W[i,i]) * XB[i, :])
        end
        G = -1/n * X' * (Z - XB) - ρ * (P - B)

        @testset "Residuals" begin
            MVDA.__evaluate_residuals__(prob, ϵ, extras, true, true, true)
            @test all(j -> res.main.dim[j] ≈ 1/sqrt(n) * view(R1, :, j), 1:c-1)
            @test all(j -> res.dist.dim[j] ≈ view(R2, :, j), 1:c-1)
            @test all(j -> res.weighted.dim[j] ≈ view(R1_scaled, :, j), 1:c-1)
            @test all(i -> view(extras.Z, i, :) ≈ view(Z, i, :), 1:n)
        end
    
        @testset "Gradient" begin
            MVDA.__evaluate_gradient__(prob, ρ, extras)
            @test all(j -> grad.dim[j] ≈ view(G, :, j), 1:c-1)
        end
    
        @testset "Objective" begin
            loss, obj, dist, gradsq = MVDA.__evaluate_objective__(prob, ϵ, ρ, extras)
            @test loss ≈ dot(R1_scaled, R1_scaled)
            @test obj ≈ 1//2 * (dot(R1_scaled, R1_scaled) + ρ * dot(R2, R2))
            @test dist ≈ dot(R2, R2)
            @test gradsq ≈ dot(G, G)
        end
    end
    
    @testset "Descent Property" begin
        ϵ = 0.5 * sqrt(2*c/(c-1))
        ρ = 1.234
        
        @testset "$(algorithm)" for algorithm in (MMSVD(), SD(),)
            function test_descent_property(s, threshold)
                B = prob.coeff.all
                rng = StableRNG(1903)
                randn!(rng, B)
                copyto!(prob.coeff_prev.all, B)
                
                _, _, obj0A, _, _ = MVDA.anneal(algorithm, prob, ϵ, ρ, s, ninner=0, gtol=1e-6, nesterov_threshold=threshold)
                _, _, obj0B, _, _ = MVDA.anneal(algorithm, prob, ϵ, ρ, s, ninner=0, gtol=1e-6, nesterov_threshold=threshold)
                _, _, obj1, _, _ = MVDA.anneal(algorithm, prob, ϵ, ρ, s, ninner=1, gtol=1e-6, nesterov_threshold=threshold)
                _, _, obj2, _, _ = MVDA.anneal(algorithm, prob, ϵ, ρ, s, ninner=1, gtol=1e-6, nesterov_threshold=threshold)
                _, _, obj100, _, _ = MVDA.anneal(algorithm, prob, ϵ, ρ, s, ninner=98, gtol=1e-6, nesterov_threshold=threshold)
                _, _, objfinal, _, gradsq = MVDA.anneal(algorithm, prob, ϵ, ρ, s, ninner=10^4, gtol=1e-8, nesterov_threshold=threshold)
    
                @test obj0A == obj0B # no iterations
                @test obj0A > obj1   # decrease after 1 iteration
                @test obj1 > obj2    # decrease after 1 iteration
                @test obj0A > obj100 # decrease after 100 iterations
                @test obj0A > objfinal # decrease at final estimate
                @test gradsq < 1e-8 # convergence
                @test all(!isnan, B) # no instability
            end
            
            # check for different model sizes
            for s in (0.0, 0.25, 0.5, 0.75)
                # w/o Nesterov acceleration
                test_descent_property(s, 100)
    
                # w/ Nesterov acceleration
                test_descent_property(s, 10)
            end
        end
    end
end    

# tests on example datasets
df = MVDA.dataset("iris")
target, X = Vector(df.target), Matrix(df[!,2:end])
k = 2

@testset "w/ Intercept" begin
    prob = MVDAProblem(target, X, intercept=true)
    test_on_dataset(prob, target, X, k)
end

@testset "w/o Intercept" begin
    prob = MVDAProblem(target, X, intercept=false)
    test_on_dataset(prob, target, X, k)
end
