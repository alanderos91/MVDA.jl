using MVDA
using MVDA: LassoPenalty
using LinearAlgebra, Random, StableRNGs, Parameters, Statistics
using Test

isless_or_approx_equal(x, y) = x < y || x ≈ y

@testset "Projections" begin
    rng = StableRNG(2000)
    @testset "L0Projection" begin
        # randomized
        for _ in 1:10
            number_components = 1000
            x = 10 * randn(number_components)
            k = 50
            P = L0Projection(rng, number_components)
            xproj = P(copy(x), k)

            nzidx = findall(xi -> xi != 0, xproj)
            @test length(nzidx) ≤ k
            @test sort!(xproj[nzidx], by=abs, rev=true) == sort(x, by=abs, rev=true)[1:k]
        end

        # ties
        x = Float64[2, 1, 1, 1, 1, -0.6, 0.5, 0.5]
        shuffle!(StableRNG(1234), x)
        P = L0Projection(rng, length(x))
        x_sorted = sort(x, lt=isless, by=abs, rev=true, order=Base.Order.Forward)
        x_zero = P(copy(x), 0)
        @test x_zero == zeros(length(x))
        for k in 1:length(x)
            x_proj = P(copy(x), k)
            idx = findall(!isequal(0), x_proj)
            topk = sort!(x_proj[idx], lt=isless, by=abs, rev=true)
            @test topk == x_sorted[1:k]
        end
    end

    @testset "HomogeneousL0Projection" begin
        ncategories = 8
        nfeatures = 100
        X = 10 * randn(nfeatures, ncategories)
        k = 10
        P = HomogeneousL0Projection(rng, nfeatures)
        Xproj = P(copy(X), k)

        norms = map(norm, eachrow(Xproj))
        nzidx = findall(!isequal(0), norms)
        @test length(nzidx) ≤ k
        @test sort!(norms[nzidx], rev=true) == sort(norms, rev=true)[1:k]
    end

    @testset "HeterogeneousL0Projection" begin
        ncategories = 8
        nfeatures = 100
        X = 10 * randn(nfeatures, ncategories)
        k = 10
        P = HeterogeneousL0Projection(rng, ncategories, nfeatures)
        Xproj = P(copy(X), k)

        correct_length, correct_subsets = true, true
        for (x, xproj) in zip(eachcol(X), eachcol(Xproj))
            nzidx = findall(xi -> xi != 0, xproj)
            correct_length = length(nzidx) ≤ k && correct_length
            correct_subsets = sort!(xproj[nzidx], by=abs, rev=true) == sort(x, by=abs, rev=true)[1:k] && correct_subsets
        end
        @test correct_length
        @test correct_subsets
    end

    @testset "L1BallProjection" begin
        number_categories = 3
        number_components = 1000
        lambda = 1.0

        # vector
        x = 10 * randn(number_components)
        P = L1BallProjection(number_components)
        xproj = P(copy(x), lambda)
        @test isless_or_approx_equal(norm(xproj, 1), lambda)

        # matrix, interpreted as vector
        x = 10 * randn(number_components, number_categories)
        lambda = 1.0
        P = L1BallProjection(number_components * number_categories)
        xproj = P(copy(x), lambda)
        @test isless_or_approx_equal(norm(xproj, 1), lambda)
    end

    @testset "HomogeneousL1BallProjection" begin
        number_categories = 3
        number_components = 1000
        lambda = 1.0

        x = 10 * randn(number_components, number_categories)
        P = HomogeneousL1BallProjection(number_categories)
        xproj = P(copy(x), lambda)
        norms = map(Base.Fix2(norm, 1), eachrow(xproj))
        @test all(Base.Fix2(isless_or_approx_equal, lambda), norms)
    end

    @testset "HeterogeneousL1BallProjection" begin
        number_categories = 3
        number_components = 1000
        lambda = 1.0

        x = 10 * randn(number_components, number_categories)
        P = HeterogeneousL1BallProjection(number_components)
        xproj = P(copy(x), lambda)
        norms = map(Base.Fix2(norm, 1), eachcol(xproj))
        @test all(Base.Fix2(isless_or_approx_equal, lambda), norms)
    end

    @testset "L2BallProjection" begin
        number_categories = 3
        number_components = 1000
        lambda = 1.0

        # vector
        x = 10 * randn(number_components)
        P = L2BallProjection()
        xproj = P(copy(x), lambda)
        @test isless_or_approx_equal(norm(xproj, 2), lambda)

        # matrix, interpreted as vector
        x = 10 * randn(number_components, number_categories)
        lambda = 1.0
        P = L2BallProjection()
        xproj = P(copy(x), lambda)
        @test isless_or_approx_equal(norm(xproj, 2), lambda)
    end

    @testset "HomogeneousL2BallProjection" begin
        number_categories = 3
        number_components = 1000
        lambda = 1.0

        x = 10 * randn(number_components, number_categories)
        P = HomogeneousL2BallProjection()
        xproj = P(copy(x), lambda)
        norms = map(Base.Fix2(norm, 2), eachrow(xproj))
        @test all(Base.Fix2(isless_or_approx_equal, lambda), norms)
    end

    @testset "HeterogeneousL2BallProjection" begin
        number_categories = 3
        number_components = 1000
        lambda = 1.0

        x = 10 * randn(number_components, number_categories)
        P = HeterogeneousL2BallProjection()
        xproj = P(copy(x), lambda)
        norms = map(Base.Fix2(norm, 2), eachcol(xproj))
        @test all(Base.Fix2(isless_or_approx_equal, lambda), norms)
    end
end

@testset "VDA models" begin
    n_samples, n_features, n_dims = 100, 10, 3
    hparams = (;
        epsilon = 0.5,
        lambda = 12.3,
        alpha = 0.1,
        rho = 8.125,
        k=round(Int, 0.5*n_features), 
    )
    A = randn(n_samples, n_features)
    B = (; slope=randn(n_features, n_dims), intercept=randn(n_dims))
    G = (; slope=zeros(n_features, n_dims), intercept=zeros(n_dims))
    R = (; loss=randn(n_samples, n_dims), dist=zeros(n_features, n_dims))

    @testset "RidgePenalty" begin
        foreach(x -> fill!(x, 0), G)
        correct_scale_factor = 1 / n_features
        expected_slope = hparams.lambda * correct_scale_factor * copy(B.slope)

        scale_factor = MVDA.get_scale_factor(RidgePenalty(), B.slope, hparams)
        penalty = MVDA.evaluate_model!(RidgePenalty(), (correct_scale_factor, B, G), hparams)

        @test scale_factor ≈ correct_scale_factor
        @test penalty ≈ dot(B.slope, B.slope) * correct_scale_factor
        @test G.slope ≈ expected_slope
        @test all(sign.(G.slope) .== sign.(expected_slope))
        @test norm(G.intercept) ≈ 0.0
    end

    @testset "LassoPenalty" begin
        foreach(x -> fill!(x, 0), G)
        correct_scale_factor = 1 / (n_features * n_dims)
        scaled_lambda = hparams.lambda * correct_scale_factor
        expected_slope = ifelse.(B.slope .< 0, -scaled_lambda, scaled_lambda)

        scale_factor = MVDA.get_scale_factor(LassoPenalty(), B.slope, hparams)
        penalty = MVDA.evaluate_model!(LassoPenalty(), (correct_scale_factor, B, G), hparams)

        @test scale_factor ≈ correct_scale_factor
        @test penalty ≈ norm(B.slope, 1) * correct_scale_factor
        @test G.slope ≈ expected_slope
        @test all(sign.(G.slope) .== sign.(expected_slope))
        @test norm(G.intercept) ≈ 0.0
    end

    # @testset "ElasticNetPenalty" begin
    #     scale_factor = MVDA.get_scale_factor(ElasticNetPenalty(), B.slope, hparams)
    #     penalty = MVDA.evaluate_model!(ElasticNetPenalty(), (scale_factor, B, G), hparams)

    #     @test scale_factor ≈ 1 / (n_features * n_dims)
    #     @test penalty ≈ norm(B, 1) / (n_features * n_dims)
    # end

    @testset "SquaredDistancePenalty" begin
        rng = StableRNG(2000)

        @testset "L0Projection" begin
            foreach(x -> fill!(x, 0), G)
            fill!(R.dist, 0)

            correct_scale_factor = 1 / (n_features * n_dims - hparams.k)
            scaled_rho = hparams.rho * correct_scale_factor
            idx = randperm(n_features * n_dims)[1:hparams.k]
            randn!(R.dist[idx])
            expected_slope = -scaled_rho * copy(R.dist)
            projection = MVDA.make_projection(L0Projection, rng, n_features, n_dims)

            scale_factor = MVDA.get_scale_factor(projection, B.slope, hparams)
            penalty = MVDA.evaluate_model!(SquaredDistancePenalty(), (scale_factor, G, R), hparams)
    
            @test scale_factor ≈ correct_scale_factor
            @test penalty ≈ dot(R.dist, R.dist) * correct_scale_factor
            @test G.slope ≈ expected_slope
            @test all(sign.(G.slope) .== sign.(expected_slope))
            @test norm(G.intercept) ≈ 0.0
        end

        @testset "HomogeneousL0Projection" begin
            foreach(x -> fill!(x, 0), G)
            fill!(R.dist, 0)

            correct_scale_factor = 1 / (n_dims * (n_features - hparams.k))
            scaled_rho = hparams.rho * correct_scale_factor
            idx = randperm(n_features)[1:hparams.k]
            randn!(R.dist[idx, :])
            expected_slope = -scaled_rho * copy(R.dist)
            projection = MVDA.make_projection(HomogeneousL0Projection, rng, n_features, n_dims)

            scale_factor = MVDA.get_scale_factor(projection, B.slope, hparams)
            penalty = MVDA.evaluate_model!(SquaredDistancePenalty(), (scale_factor, G, R), hparams)
    
            @test scale_factor ≈ correct_scale_factor
            @test penalty ≈ dot(R.dist, R.dist) * correct_scale_factor
            @test G.slope ≈ expected_slope
            @test all(sign.(G.slope) .== sign.(expected_slope))
            @test norm(G.intercept) ≈ 0.0
        end

        @testset "HeterogeneousL0Projection" begin
            foreach(x -> fill!(x, 0), G)
            fill!(R.dist, 0)

            correct_scale_factor = 1 / (n_dims * (n_features - hparams.k))
            scaled_rho = hparams.rho * correct_scale_factor
            for l in axes(R.dist, 2)
                idx = randperm(n_features)[1:hparams.k]
                randn!(R.dist[idx, l])
            end
            expected_slope = -scaled_rho * copy(R.dist)
            projection = MVDA.make_projection(HeterogeneousL0Projection, rng, n_features, n_dims)

            scale_factor = MVDA.get_scale_factor(projection, B.slope, hparams)
            penalty = MVDA.evaluate_model!(SquaredDistancePenalty(), (scale_factor, G, R), hparams)
    
            @test scale_factor ≈ correct_scale_factor
            @test penalty ≈ dot(R.dist, R.dist) * correct_scale_factor
            @test G.slope ≈ expected_slope
            @test all(sign.(G.slope) .== sign.(expected_slope))
            @test norm(G.intercept) ≈ 0.0
        end

        @testset "L1BallProjection" begin
            foreach(x -> fill!(x, 0), G)
            randn!(R.dist)

            correct_scale_factor = 1 / (n_features * n_dims)
            scaled_rho = hparams.rho * correct_scale_factor
            expected_slope = -scaled_rho * copy(R.dist)
            projection = MVDA.make_projection(L1BallProjection, rng, n_features, n_dims)

            scale_factor = MVDA.get_scale_factor(projection, B.slope, hparams)
            penalty = MVDA.evaluate_model!(SquaredDistancePenalty(), (scale_factor, G, R), hparams)
    
            @test scale_factor ≈ correct_scale_factor
            @test penalty ≈ dot(R.dist, R.dist) * correct_scale_factor
            @test G.slope ≈ expected_slope
            @test all(sign.(G.slope) .== sign.(expected_slope))
            @test norm(G.intercept) ≈ 0.0
        end

        @testset "HomogeneousL1BallProjection" begin
            foreach(x -> fill!(x, 0), G)
            randn!(R.dist)

            correct_scale_factor = 1 / (n_features * n_dims)
            scaled_rho = hparams.rho * correct_scale_factor
            expected_slope = -scaled_rho * copy(R.dist)
            projection = MVDA.make_projection(HomogeneousL1BallProjection, rng, n_features, n_dims)

            scale_factor = MVDA.get_scale_factor(projection, B.slope, hparams)
            penalty = MVDA.evaluate_model!(SquaredDistancePenalty(), (scale_factor, G, R), hparams)
    
            @test scale_factor ≈ correct_scale_factor
            @test penalty ≈ dot(R.dist, R.dist) * correct_scale_factor
            @test G.slope ≈ expected_slope
            @test all(sign.(G.slope) .== sign.(expected_slope))
            @test norm(G.intercept) ≈ 0.0
        end

        @testset "HeterogeneousL1BallProjection" begin
            foreach(x -> fill!(x, 0), G)
            randn!(R.dist)

            correct_scale_factor = 1 / (n_features * n_dims)
            scaled_rho = hparams.rho * correct_scale_factor
            expected_slope = -scaled_rho * copy(R.dist)
            projection = MVDA.make_projection(HeterogeneousL1BallProjection, rng, n_features, n_dims)

            scale_factor = MVDA.get_scale_factor(projection, B.slope, hparams)
            penalty = MVDA.evaluate_model!(SquaredDistancePenalty(), (scale_factor, G, R), hparams)
    
            @test scale_factor ≈ correct_scale_factor
            @test penalty ≈ dot(R.dist, R.dist) * correct_scale_factor
            @test G.slope ≈ expected_slope
            @test all(sign.(G.slope) .== sign.(expected_slope))
            @test norm(G.intercept) ≈ 0.0
        end

        @testset "L2BallProjection" begin
            foreach(x -> fill!(x, 0), G)
            randn!(R.dist)

            correct_scale_factor = 1 / (n_features * n_dims)
            scaled_rho = hparams.rho * correct_scale_factor
            expected_slope = -scaled_rho * copy(R.dist)
            projection = MVDA.make_projection(L2BallProjection, rng, n_features, n_dims)

            scale_factor = MVDA.get_scale_factor(projection, B.slope, hparams)
            penalty = MVDA.evaluate_model!(SquaredDistancePenalty(), (scale_factor, G, R), hparams)
    
            @test scale_factor ≈ correct_scale_factor
            @test penalty ≈ dot(R.dist, R.dist) * correct_scale_factor
            @test G.slope ≈ expected_slope
            @test all(sign.(G.slope) .== sign.(expected_slope))
            @test norm(G.intercept) ≈ 0.0
        end

        @testset "HomogeneousL2BallProjection" begin
            foreach(x -> fill!(x, 0), G)
            randn!(R.dist)

            correct_scale_factor = 1 / (n_features * n_dims)
            scaled_rho = hparams.rho * correct_scale_factor
            expected_slope = -scaled_rho * copy(R.dist)
            projection = MVDA.make_projection(HomogeneousL2BallProjection, rng, n_features, n_dims)

            scale_factor = MVDA.get_scale_factor(projection, B.slope, hparams)
            penalty = MVDA.evaluate_model!(SquaredDistancePenalty(), (scale_factor, G, R), hparams)
    
            @test scale_factor ≈ correct_scale_factor
            @test penalty ≈ dot(R.dist, R.dist) * correct_scale_factor
            @test G.slope ≈ expected_slope
            @test all(sign.(G.slope) .== sign.(expected_slope))
            @test norm(G.intercept) ≈ 0.0
        end

        @testset "HeterogeneousL2BallProjection" begin
            foreach(x -> fill!(x, 0), G)
            randn!(R.dist)

            correct_scale_factor = 1 / (n_features * n_dims)
            scaled_rho = hparams.rho * correct_scale_factor
            expected_slope = -scaled_rho * copy(R.dist)
            projection = MVDA.make_projection(HeterogeneousL2BallProjection, rng, n_features, n_dims)

            scale_factor = MVDA.get_scale_factor(projection, B.slope, hparams)
            penalty = MVDA.evaluate_model!(SquaredDistancePenalty(), (scale_factor, G, R), hparams)
    
            @test scale_factor ≈ correct_scale_factor
            @test penalty ≈ dot(R.dist, R.dist) * correct_scale_factor
            @test G.slope ≈ expected_slope
            @test all(sign.(G.slope) .== sign.(expected_slope))
            @test norm(G.intercept) ≈ 0.0
        end
    end

    @testset "SquaredEpsilonInsensitiveLoss" begin
        alpha, beta = 1 / n_samples, 0.0

        # use large nonsense values to verify that evaluation uses buffers correctly
        fill!(R.dist, 1e3)
        foreach(x -> fill!(x, 1e3), G)
        expected = (;
            slope=-alpha * transpose(A) * R.loss,
            intercept=-vec(mean(R.loss, dims=1)),
        )

        loss = MVDA.evaluate_model!(SquaredEpsilonInsensitiveLoss(), (alpha, beta, A, G, R, true), hparams)

        @test loss ≈ dot(R.loss, R.loss) * alpha
        @test G.slope ≈ expected.slope
        @test G.intercept ≈ expected.intercept
        @test all(sign.(G.slope) .== sign.(expected.slope))
        @test all(sign.(G.intercept) .== sign.(expected.intercept))

        fill!(R.dist, 1e3)
        foreach(x -> fill!(x, 1e3), G)
        expected = (;
            slope=-alpha * transpose(A) * R.loss,
            intercept=zeros(n_dims),
        )

        loss = MVDA.evaluate_model!(SquaredEpsilonInsensitiveLoss(), (alpha, beta, A, G, R, false), hparams)

        @test loss ≈ dot(R.loss, R.loss) * alpha
        @test G.slope ≈ expected.slope
        @test G.intercept ≈ expected.intercept
        @test all(sign.(G.slope) .== sign.(expected.slope))
        @test all(sign.(G.intercept) .== sign.(expected.intercept))
    end
end

function test_on_dataset(prob, L, X, k)
    n, p, c = length(L), size(X, 2), length(unique(L))
    enc = prob.encoding
    if enc isa MVDA.StandardSimplexEncoding
        nd = c
    elseif enc isa MVDA.ProjectedSimplexEncoding
        nd = c-1
    end

    @testset "MVDAProblem" begin
        # coefficient shape
        for field in (:coeff, :coeff_prev, :coeff_proj, :grad)
            arr = getfield(prob, field)
            @test size(arr.slope, 1) == p && size(arr.slope, 2) == nd
            @test length(arr.intercept) == nd
        end

        # residual shape
        r1 = prob.res.loss
        r2 = prob.res.dist
        @test size(r1, 1) == n && size(r1, 2) == nd
        @test size(r2, 1) == p && size(r2, 2) == nd

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

    @testset "Descent Property" begin
        hparams = (
            epsilon=0.5,
            lambda=1.0,
            rho=1.234,
        )

        loss_models = (SquaredEpsilonInsensitiveLoss(),)
        penalties = (RidgePenalty(),)
        @testset "$(loss) + $(penalty)" for loss in loss_models, penalty in penalties
            f = PenalizedObjective(loss, penalty)
            @testset "$(algorithm)" for algorithm in (MMSVD(), SD(),)
                function test_descent_property(hparams, threshold)
                    B = prob.coeff.slope
                    rng = StableRNG(1903)
                    randn!(rng, B)
                    copyto!(prob.coeff_prev.slope, B)

                    (_, state0A), _ = MVDA.solve_unconstrained!(f, algorithm, prob, hparams, maxiter=0, gtol=1e-3, nesterov=threshold)
                    (_, state0B), _ = MVDA.solve_unconstrained!(f, algorithm, prob, hparams, maxiter=0, gtol=1e-3, nesterov=threshold)
                    (_, state1), _ = MVDA.solve_unconstrained!(f, algorithm, prob, hparams, maxiter=1, gtol=1e-3, nesterov=threshold)
                    (_, state2), _ = MVDA.solve_unconstrained!(f, algorithm, prob, hparams, maxiter=1, gtol=1e-3, nesterov=threshold)
                    (_, state100), _ = MVDA.solve_unconstrained!(f, algorithm, prob, hparams, maxiter=98, gtol=1e-3, nesterov=threshold)
                    (_, statefinal), _ = MVDA.solve_unconstrained!(f, algorithm, prob, hparams, maxiter=10^4, gtol=1e-4, nesterov=threshold)

                    @test state0A.objective == state0B.objective # no iterations
                    @test state0A.objective > state1.objective   # decrease after 1 iteration
                    @test state1.objective > state2.objective    # decrease after 1 iteration
                    @test state0A.objective > state100.objective # decrease after 100 iterations
                    @test state0A.objective > statefinal.objective # decrease at final estimate
                    @test statefinal.gradient < 1e-4 # convergence
                    @test all(!isnan, B) # no instability
                end

                # w/o Nesterov acceleration
                test_descent_property(hparams, 100)

                # w/ Nesterov acceleration
                test_descent_property(hparams, 10)
            end
        end

        f = PenalizedObjective(SquaredEpsilonInsensitiveLoss(), SquaredDistancePenalty())
        projections = (
            L0Projection, HomogeneousL0Projection, HeterogeneousL0Projection,
            L1BallProjection, HomogeneousL1BallProjection, HeterogeneousL1BallProjection,
            L2BallProjection, HomogeneousL2BallProjection, HeterogeneousL2BallProjection,
        )
        @testset "$(loss) + $(projection)" for loss in loss_models, projection in projections
            @testset "$(algorithm)" for algorithm in (MMSVD(), SD(),)
                function test_descent_property(hparams, threshold)
                    B = prob.coeff.slope
                    rng = StableRNG(1903)
                    randn!(rng, B)
                    copyto!(prob.coeff_prev.slope, B)

                    (_, state0A), _ = MVDA.solve_unconstrained!(f, algorithm, prob, hparams, maxiter=0, gtol=1e-3, nesterov=threshold, projection_type=projection, rng=rng)
                    (_, state0B), _ = MVDA.solve_unconstrained!(f, algorithm, prob, hparams, maxiter=0, gtol=1e-3, nesterov=threshold, projection_type=projection, rng=rng)
                    (_, state1), _ = MVDA.solve_unconstrained!(f, algorithm, prob, hparams, maxiter=1, gtol=1e-3, nesterov=threshold, projection_type=projection, rng=rng)
                    (_, state2), _ = MVDA.solve_unconstrained!(f, algorithm, prob, hparams, maxiter=1, gtol=1e-3, nesterov=threshold, projection_type=projection, rng=rng)
                    (_, state100), _ = MVDA.solve_unconstrained!(f, algorithm, prob, hparams, maxiter=98, gtol=1e-3, nesterov=threshold, projection_type=projection, rng=rng)
                    (_, statefinal), _ = MVDA.solve_unconstrained!(f, algorithm, prob, hparams, maxiter=10^4, gtol=1e-4, nesterov=threshold, projection_type=projection, rng=rng)

                    @test state0A.objective == state0B.objective # no iterations
                    @test state0A.objective > state1.objective   # decrease after 1 iteration
                    @test state1.objective > state2.objective    # decrease after 1 iteration
                    @test state0A.objective > state100.objective # decrease after 100 iterations
                    @test state0A.objective > statefinal.objective # decrease at final estimate
                    @test statefinal.gradient < 1e-4 # convergence
                    @test all(!isnan, B) # no instability
                end

                if projection <: L0Projection || projection <: HomogeneousL0Projection || projection <: HeterogeneousL0Projection
                    # check for different model sizes
                    for s in (0.0, 0.25, 0.5, 0.75)
                        k = MVDA.sparsity_to_k(prob, s)
                        _hparams = (; hparams..., k=k,)

                        # w/o Nesterov acceleration
                        test_descent_property(_hparams, 100)

                        # w/ Nesterov acceleration
                        test_descent_property(_hparams, 10)
                    end
                else
                    # w/o Nesterov acceleration
                    test_descent_property(hparams, 100)

                    # w/ Nesterov acceleration
                    test_descent_property(hparams, 10)
                end
            end
        end
    end
end    

# tests on example datasets
df = MVDA.dataset("iris")
L, X = Vector{String}(df[!,1]), Matrix{Float64}(df[!,2:end])
k = 2

@testset "ProjectedSimplexEncoding" begin
    @testset "w/ Intercept" begin
        prob = MVDAProblem(L, X, encoding=:projected, intercept=true)
        test_on_dataset(prob, L, X, k)
    end

    @testset "w/o Intercept" begin
        prob = MVDAProblem(L, X, encoding=:projected, intercept=false)
        test_on_dataset(prob, L, X, k)
    end
end

@testset "StandardSimplexEncoding" begin
    @testset "w/ Intercept" begin
        prob = MVDAProblem(L, X, encoding=:standard, intercept=true)
        test_on_dataset(prob, L, X, k)
    end

    @testset "w/o Intercept" begin
        prob = MVDAProblem(L, X, encoding=:standard, intercept=false)
        test_on_dataset(prob, L, X, k)
    end
end

# using Literate

# Literate.markdown(
#     joinpath(@__DIR__, "benchmarking.jl"),
#     joinpath(@__DIR__, "output");
#     execute=true,
#     flavor=Literate.CommonMarkFlavor()
# )