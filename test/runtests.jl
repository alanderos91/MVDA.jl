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

function test_on_dataset(prob, L, X, k)
    n, p, c = length(L), size(X, 2), length(unique(L))

    @testset "MVDAProblem" begin
        # coefficient shape
        for field in (:coeff, :coeff_prev, :coeff_proj, :grad)
            arr = getfield(prob, field)
            @test size(arr.slope, 1) == p && size(arr.slope, 2) == c-1
            @test length(arr.intercept) == c-1
        end
    
        # residual shape
        r1 = prob.res.loss
        r2 = prob.res.dist
        @test size(r1, 1) == n && size(r1, 2) == c-1
        @test size(r2, 1) == p && size(r2, 2) == c-1
    
        # verify problem dimensions
        dims = MVDA.probdims(prob)
        @test n == dims[1] && p == dims[2] && c == dims[3]
    
        # check vertex definitions
        v = prob.encoding.vertex
        d = [norm(v[j] - v[i]) for i in 1:c for j in i+1:c]
        dmin, dmax = extrema(d)
        @test all(j -> norm(v[j]) ≈ 1, eachindex(v))    # vertices lie on unit circle
        @test dmin ≈ dmax                               # vertices are equidistant
    end
    
    @testset "Evaluation" begin
        @unpack Y, X, coeff, res, coeff_proj, grad = prob
        B = coeff.slope
        P = coeff_proj.slope
        T = Float64
        has_intercept = prob.intercept
    
        # Initialize coefficients and parameters
        rng = StableRNG(1903)
        randn!(rng, B)
        ϵ = 0.5 * sqrt(2*c/(c-1))
        λ = 1.0
        ρ = 4.7
    
        # Initialize other data structures.
        extras = MVDA.__mm_init__(MMSVD(), prob, nothing)
        MVDA.__mm_update_rho__(MMSVD(), prob, extras, λ, ρ)

        # Set projection.
        MVDA.apply_projection(extras.projection, prob, k)

        # Compute residuals and gradient.
        XB = X*B
        R1 = Y - XB
        R2 = P - B

        normr = [norm(ri) for ri in eachrow(R1)]
        W = Diagonal( [ifelse(normr[i] ≤ ϵ, zero(T), (normr[i]-ϵ) / normr[i]) for i in eachindex(normr)] )
        Z = similar(Y)
        @views for i in axes(Z, 1)
            Z[i, :] .= ifelse(normr[i] ≤ ϵ, XB[i, :], W[i,i] * Y[i, :] + (1-W[i,i]) * XB[i, :])
            R1[i, :] .= Z[i, :] - XB[i, :]
        end
        G = -1/n * X' * (Z - XB) - ρ/p * (P - B) + λ/p * B
        g = if has_intercept
            -vec(mean(R1, dims=1))
        else
            zeros(c-1)
        end
        @testset "Residuals" begin
            MVDA.evaluate_residuals!(prob, extras, ϵ, true, true)
            @test norm(res.loss .- R1) < sqrt(eps())
            @test norm(res.dist .- R2) < sqrt(eps())
            @test norm(extras.Z .- Z) < sqrt(eps())
        end
    
        @testset "Gradient" begin
            MVDA.evaluate_gradient!(prob, λ, ρ)
            @test norm(grad.slope .- G) < sqrt(eps())
        end
    
        @testset "Objective" begin
            nt = MVDA.evaluate_objective!(prob, extras, ϵ, λ, ρ)
            @test nt.risk ≈ 1//n * dot(R1, R1)
            @test nt.loss ≈ 1//2 * (1//n * dot(R1, R1) + λ/p * dot(B, B))
            @test nt.objective ≈ 1//2 * (1//n * dot(R1, R1) + λ/p * dot(B, B) + ρ/p * dot(R2, R2))
            @test nt.distance ≈ sqrt(dot(R2, R2))
            @test nt.gradient ≈ sqrt(dot(G, G) + dot(g, g))
        end
    end
    
    @testset "Descent Property" begin
        ϵ = 0.5 * sqrt(2*c/(c-1))
        λ = 1.0
        ρ = 1.234
        
        @testset "$(algorithm)" for algorithm in (MMSVD(), SD(),)
            function test_descent_property(s, threshold)
                B = prob.coeff.slope
                rng = StableRNG(1903)
                randn!(rng, B)
                copyto!(prob.coeff_prev.slope, B)
                
                (_, state0A) = MVDA.anneal!(algorithm, prob, ϵ, λ, ρ, s, maxiter=0, gtol=1e-3, nesterov=threshold)
                (_, state0B) = MVDA.anneal!(algorithm, prob, ϵ, λ, ρ, s, maxiter=0, gtol=1e-3, nesterov=threshold)
                (_, state1) = MVDA.anneal!(algorithm, prob, ϵ, λ, ρ, s, maxiter=1, gtol=1e-3, nesterov=threshold)
                (_, state2) = MVDA.anneal!(algorithm, prob, ϵ, λ, ρ, s, maxiter=1, gtol=1e-3, nesterov=threshold)
                (_, state100) = MVDA.anneal!(algorithm, prob, ϵ, λ, ρ, s, maxiter=98, gtol=1e-3, nesterov=threshold)
                (_, statefinal) = MVDA.anneal!(algorithm, prob, ϵ, λ, ρ, s, maxiter=10^4, gtol=1e-4, nesterov=threshold)
    
                @test state0A.objective == state0B.objective # no iterations
                @test state0A.objective > state1.objective   # decrease after 1 iteration
                @test state1.objective > state2.objective    # decrease after 1 iteration
                @test state0A.objective > state100.objective # decrease after 100 iterations
                @test state0A.objective > statefinal.objective # decrease at final estimate
                @test statefinal.gradient < 1e-4 # convergence
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
L, X = Vector(df[!,1]), Matrix(df[!,2:end])
k = 2

@testset "w/ Intercept" begin
    prob = MVDAProblem(L, X, intercept=true)
    test_on_dataset(prob, L, X, k)
end

@testset "w/o Intercept" begin
    prob = MVDAProblem(L, X, intercept=false)
    test_on_dataset(prob, L, X, k)
end
