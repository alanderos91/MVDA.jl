using RCall
R"""
library(sparseLDA)
library(MGSDA)
"""

import LIBSVM
import LIBSVM: LinearSVC, Linearsolver
import MVDA: accuracy

abstract type AbstractWrapper end

function MVDA.accuracy(model::AbstractWrapper, (L, X))
    sum(MLDataUtils.predict(model, X) .== L) / length(L)
end

struct MGSDAWrapper <: AbstractWrapper
    X::Matrix{Float64}
    Y::Vector{Int}
    V::Matrix{Float64}
    labels::Vector{String}
    lambda::Float64
    eps::Float64
    maxiter::Int
end

function fit_MGSDA(L, X; lambda::Float64=1.0, eps::Float64=1e-6, maxiter::Int=1000, kwargs...)
    @rput lambda eps maxiter
    R"""
    Lfactor <- factor($L)
    labels <- levels(Lfactor)
    Y <- as.double(Lfactor)
    V <- dLDA($X, Y, lambda=lambda, eps=eps, maxiter=maxiter)
    Y <- as.vector(Y)
    """
    @rget Y V labels
    return MGSDAWrapper(X, Y, V, labels, lambda, eps, maxiter)
end

function cv_MGSDA(L, X; nlambda::Int=10, eps::Float64=1e-6, nfolds::Int=5, seed::Int=0, kwargs...)
    @rput nlambda eps nfolds seed
    R"""
    Y <- as.double(factor($L))
    result <- cv.dLDA($X, Y, eps=eps, msep=nfolds, nl=nlambda, myseed=seed)
    lambda <- result$lambda_min
    """
    println()
    @rget lambda
    return lambda
end

function MGSDA(L, X; at::Float64=0.8, kwargs...)
    data = MLDataUtils.splitobs((L, X), at=at, obsdim=ObsDim.First())
    L_cv, X_cv = MLDataUtils.getobs(data[1], ObsDim.First())
    L_ts, X_ts = MLDataUtils.getobs(data[2], ObsDim.First())
    @info "MGSDA" n_train=length(L_cv) n_test=length(L_ts)
    result_cv = @timed cv_MGSDA(L_cv, X_cv; kwargs...)
    result_fit = @timed fit_MGSDA(L_cv, X_cv; lambda=result_cv.value, kwargs...)
    model = result_fit.value
    idx = findall(!=(0), map(sum, eachrow(model.V)))
    acc_tr = MVDA.accuracy(model, (L_cv, X_cv))
    acc_ts = MVDA.accuracy(model, (L_ts, X_ts))
    return (
        hyperparameters=(lambda=result_cv.value,),
        path=(
            result=(time=result_cv.time,),
        ),
        fit=(
            time=result_fit.time,
            train=(score=acc_tr,),
            test=(score=acc_ts,),
            support=idx,
        ),
        reduced=(
            time=0.0,
            train=(score=acc_tr,),
            test=(score=acc_ts,),
        ),
    )
end

function MLDataUtils.predict(model::MGSDAWrapper, newX=model.X)
    X, Y, V = model.X, model.Y, model.V
    class_idx = rcopy(R"as.integer(classifyV($X, $Y, $newX, $V))")
    return model.labels[class_idx]
end

struct SparseLDAWrapper <: AbstractWrapper
    X::RObject
    fit::RObject
    k::Int
    idx::Vector{Int}
    lambda::Float64
    maxiter::Int
end

function fit_sparseLDA(L, X; k::Int=div(size(X, 2), 2), lambda=1.0, maxiter=10)
    @rput k lambda maxiter
    R"""
    X <- normalize($X)
    Y <- factor($L)
    fit <- sda(X$Xc, Y, maxIte=maxiter, stop=-k, lambda=lambda)
    idx <- fit$varIndex
    """
    @rget idx
    robjX = robject(R"X")
    robjfit = robject(R"fit")
    return SparseLDAWrapper(robjX, robjfit, k, idx, lambda, maxiter)
end

function cv_sparseLDA(L, X; nlambda::Int=5, nk::Int=10, nfolds::Int=5, maxiter::Int=10)
    data = (L, X)
    lambdas = MVDA.make_log10_grid(log10(sqrt(size(X, 2))), 0, nlambda)
    best_lambda, best_cost = 0.0, Inf
    for lambda in lambdas
        model = fit_sparseLDA(L, X, maxiter=maxiter, lambda=lambda)
        fit = model.fit
        R"""
        fit <- $fit
        cost <- sum(fit$rss) + sum(abs(fit$beta))
        """
        @rget cost
        if cost < best_cost
            best_lambda = lambda
            best_cost = cost
        end
    end
    p = size(X, 2)
    grid = unique(round.(Int, p .* (1 .- MVDA.make_sparsity_grid(p, nk))))
    acc = zeros(length(grid), nfolds)
    for (j, fold) in enumerate(kfolds(data, k=nfolds, obsdim=1))
        # Retrieve the training set and validation set.
        train_set, validation_set = fold
        train_L, train_X = MLDataUtils.getobs(train_set, obsdim=1)
        val_L, val_X = MLDataUtils.getobs(validation_set, obsdim=1)
        for (i, k) in enumerate(grid)
            model = fit_sparseLDA(train_L, train_X, k=k, maxiter=maxiter, lambda=best_lambda)
            acc[i,j] = MVDA.accuracy(model, (val_L, val_X))
        end
    end
    avg_acc = vec(mean(acc, dims=2))
    best_k = grid[findlast(==(maximum(avg_acc)), avg_acc)]
    return (lambda=best_lambda, k=best_k)
end

function sparseLDA(L, X; at::Float64=0.8, kwargs...)
    data = MLDataUtils.splitobs((L, X), at=at, obsdim=ObsDim.First())
    L_cv, X_cv = MLDataUtils.getobs(data[1], ObsDim.First())
    L_ts, X_ts = MLDataUtils.getobs(data[2], ObsDim.First())
    result_cv = @timed cv_sparseLDA(L_cv, X_cv; kwargs...)
    result_fit = @timed fit_sparseLDA(L_cv, X_cv;
        k=result_cv.value.k, lambda=result_cv.value.lambda, kwargs...)
    model = result_fit.value
    acc_tr = MVDA.accuracy(model, (L_cv, X_cv))
    acc_ts = MVDA.accuracy(model, (L_ts, X_ts))
    return (
        hyperparameters=(lambda=result_cv.value,),
        path=(
            result=(time=result_cv.time,),
        ),
        fit=(
            time=result_fit.time,
            train=(score=acc_tr,),
            test=(score=acc_ts,),
            support=model.idx,
        ),
        reduced=(
            time=0.0,
            train=(score=acc_tr,),
            test=(score=acc_ts,),
        ),
    )
end

function MLDataUtils.predict(model::SparseLDAWrapper, newX=model.X)
    X, fit = model.X, model.fit
    R"""
    newX <- normalizetest($newX, $X)
    Lhat <- as.vector(predict($fit, newX)$class)
    """
    @rget Lhat
    return Lhat
end

function MVDA.accuracy(model::LinearSVC, (L, X))
    sum(LIBSVM.predict(model, X) .== L) / length(L)
end

function fit_L1R_L2LOSS_SVC(L, X;
    tolerance::Float64=Inf,
    cost::Float64=1.0,
    bias::Float64=-1.0,
    verbose::Bool=false,
    kwargs...
)
    #
    model = LinearSVC(
        solver=Linearsolver.L1R_L2LOSS_SVC,
        tolerance=tolerance,
        cost=cost,
        bias=bias,
        verbose=verbose
    );
    return LIBSVM.fit!(model, X, L)
end

function cv_L1R_L2LOSS_SVC(L, X;
    tolerance::Float64=Inf,
    bias::Float64=-1.0,
    verbose::Bool=false,
    Cvals=MVDA.make_log10_grid(-2, 2, 5),
    nfolds::Int=5,
    seed::Int=1903,
    kwargs...
)
    # helper function to create a model
    make_model(C) = LinearSVC(
        solver=Linearsolver.L1R_L2LOSS_SVC,
        tolerance=tolerance,
        cost=C,
        bias=bias,
        verbose=false
    )
    # Search C grid from smallest to largest
    sort!(Cvals, lt=isless, rev=false)
    data = (L, X)
    acc = zeros(length(Cvals), nfolds)
    for (j, fold) in enumerate(kfolds(data, k=nfolds, obsdim=1))
        # Retrieve the training set and validation set.
        train_set, validation_set = fold
        train_L, train_X = MLDataUtils.getobs(train_set, obsdim=1)
        val_L, val_X = MLDataUtils.getobs(validation_set, obsdim=1)
        # Center and scale both datasets using only the training data
        F = StatsBase.fit(ZScoreTransform, train_X, dims=1)
        MVDA.__adjust_transform__(F)
        foreach(Base.Fix1(StatsBase.transform!, F), (train_X, val_X))
        # Find the best value of C
        for (i, C) in enumerate(Cvals)
            model = LIBSVM.fit!(make_model(C), train_X, train_L)
            acc[i,j] = MVDA.accuracy(model, (val_L, val_X))
        end
    end
    avg_acc = vec(mean(acc, dims=2))
    best_cost = Cvals[findfirst(==(maximum(avg_acc)), avg_acc)]
    return (; cost=best_cost)
end

function L1R_L2LOSS_SVC(L, X; is_class_specific::Bool=false, at::Float64=0.8, kwargs...)
    # Split data into CV and test sets
    data = MLDataUtils.splitobs((L, X), at=at, obsdim=ObsDim.First())
    L_cv, X_cv = MLDataUtils.getobs(data[1], ObsDim.First())
    L_ts, X_ts = MLDataUtils.getobs(data[2], ObsDim.First())
    @info "L1R_L2LOSS_SVC" n_train=length(L_cv) n_test=length(L_ts)
    # Run CV
    result_cv = @timed cv_L1R_L2LOSS_SVC(L_cv, X_cv; kwargs...)
    # Center and scale both datasets using only the training data
    F = StatsBase.fit(ZScoreTransform, X_cv, dims=1)
    MVDA.__adjust_transform__(F)
    foreach(Base.Fix1(StatsBase.transform!, F), (X_cv, X_ts))
    # Train the final model
    result_fit = @timed fit_L1R_L2LOSS_SVC(L_cv, X_cv;
        cost=result_cv.value.cost, kwargs...)
    model = result_fit.value
    B = reshape(model.fit.w, model.fit.nr_class, model.fit.nr_feature) |>
        transpose |> Matrix
    if is_class_specific
        idx = [findall(!=(0), b) for b in eachcol(B)]
    else
        idx = findall(!=(0), [norm(b) for b in eachrow(B)])
    end
    acc_tr = MVDA.accuracy(model, (L_cv, X_cv))
    acc_ts = MVDA.accuracy(model, (L_ts, X_ts))
    return (
        hyperparameters=(cost=result_cv.value.cost,),
        path=(
            result=(time=result_cv.time,),
        ),
        fit=(
            time=result_fit.time,
            train=(score=acc_tr,),
            test=(score=acc_ts,),
            support=idx,
        ),
        reduced=(
            time=0.0,
            train=(score=acc_tr,),
            test=(score=acc_ts,),
        ),
    )
    return model
end
