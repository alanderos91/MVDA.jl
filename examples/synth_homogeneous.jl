#
#   Synthetic Multiclass Data with Homogeneous Feature Set
#
#   Author: Alfonso Landeros
#   Reviewed: 04 / 27 / 24
#
include("common.jl")
include("wrappers.jl")

function init_dataframe()
    return DataFrame(
        # settings
        n=Int[],
        p=Int[],
        c=Int[],
        k=Int[],
        rho=Float64[],
        SNR=Float64[],
        replicate=Int[],
        solver=String[],
        # computation + model selection
        time=Float64[],
        hyperparameter=Float64[],
        # classification accuracy
        accuracy_train=Float64[],
        accuracy_test=Float64[],
        accuracy_train_debiased=Float64[],
        accuracy_test_debiased=Float64[],
        # support recovery
        active_features=Int[],
        true_features=Int[],
        false_features=Int[],
    )
end

function record_results!(df, settings, i, solver, result, true_support, param_symbol)
    n, p, c, k, rho, SNR = settings
    if haskey(result.fit, :problem)
        model = result.fit.problem
        support = findall(!=(0), [norm(b) for b in eachrow(model.coeff_proj.slope)])
    else
        support = result.fit.support
    end
    new_row = (
        # settings
        n, p, c, k, rho, SNR, i,
        solver,
        # we delibrately ignore result.tune.result.time; no tuning takes place
        sum(result.path.result.time) + result.fit.time + result.reduced.time,
        getindex(result.hyperparameters, param_symbol),
        # classification accuracy
        result.fit.train.score,
        result.fit.test.score,
        result.reduced.train.score,
        result.reduced.test.score,
        # support recovery
        length(support),
        length(intersect(support, true_support)),
        length(setdiff(support, true_support)),
    )
    push!(df, new_row)
end

function main(seed)
    #
    #   Simulation Parameters
    #
    loss_model = PenalizedObjective(SqEpsilonLoss(),SqDistPenalty())
    ns = [500, 2000]
    ps = [1000]
    cs = [3, 10]
    ks = [30]
    rhos = [0.1, 0.5, 0.9]
    SNRs = [0.1, 1.0, 10.0]
    #
    #   RNGs + Settings
    #
    simRNG = StableRNG(seed)
    repRNG = StableRNG(seed)
    gtol, dtol, rtol = 1e-3, 1e-3, 0.0
    mgsda_maxiter = 2000
    mgsda_tol = 1e-4
    mgsda_nlambda = 51
    svm_tol = 1e-3
    ntest = 1000
    nfolds = 5
    nreps = 10
    #
    #   Initialize output
    #
    dir = joinpath("$(homedir())", "Desktop", "VDA-Results", "synthetic")
    mkpath(dir)
    filename = joinpath(dir, "synth_homogeneous.csv")
    df = init_dataframe()
    CSV.write(filename, df)
    for settings in Iterators.product(ns, ps, cs, ks, rhos, SNRs)
        for i in 1:nreps
            df = init_dataframe()
            #
            # Simulate an instance
            #
            # N = Total number of instances that are split so that
            # (1) the test set has ntest instances
            # (2) each fold in CV has n instances
            #
            (n, p, c, k, rho, SNR) = settings
            N = ceil(Int, n * nfolds/(nfolds-1)) + ntest
            q = (N - ntest) / N
            _L, _, _X, B = MVDA.vdasynth1(N, p, c, k; rho=rho, SNR=SNR, rng=simRNG);
            true_support = findall(!=(0), [norm(b) for b in eachrow(B)])
            L, X = MLDataUtils.getobs(
                MLDataUtils.shuffleobs((_L, _X), ObsDim.First(), simRNG),
                ObsDim.First()
            )
            problem = MVDAProblem(L, X; intercept=false, encoding=:standard)
            #
            # Set grids for hyperparameters
            #
            grids = (
                epsilon=[MVDA.maximum_deadzone(problem)],
                lambda=sort!(MVDA.make_log10_grid(-3, 3, 100), rev=true),
                gamma=[0.0],
                k=MVDA.make_sparsity_grid(1000, 100),
            )
            #
            # Solve with L0-type MVDA
            #
            Random.seed!(repRNG, i)
            l0_result = MVDA.cv(loss_model, MMSVD(), problem, grids;
                data=MVDA.split_dataset(problem, q),
                nfolds=nfolds,
                gtol=gtol,
                dtol=dtol,
                rtol=rtol,
                projection_type=HomogeneousL0Projection,
                rng=repRNG,
            )
            record_results!(df, settings, i, "HomL0", l0_result, true_support, :k)
            #
            # Solve with L1-type MVDA
            #
            Random.seed!(repRNG, i)
            l1_result = MVDA.cv(loss_model, MMSVD(), problem, grids;
                data=MVDA.split_dataset(problem, q),
                nfolds=nfolds,
                gtol=gtol,
                dtol=dtol,
                rtol=rtol,
                projection_type=L1BallProjection,
                rng=repRNG,
            )
            record_results!(df, settings, i, "L1", l1_result, true_support, :lambda)
            #
            # Solve with MGSDA
            #
            mgsda_result = MGSDA(L, X;
                at=q,
                nlambda=mgsda_nlambda,
                maxiter=mgsda_maxiter,
                eps=mgsda_tol,
                nfolds=nfolds,
                seed=i,
            )
            record_results!(df, settings, i, "MGSDA", mgsda_result, true_support, :lambda)
            #
            # Solve with L1-regularized L2-loss SVC
            #
            grid = MVDA.make_log10_grid(-2, 0, 5)
            l1rsvm_result = L1R_L2LOSS_SVC(L, X;
                is_class_specific=false,
                at=q,
                nfolds=nfolds,
                Cvals=grid,
                bias=-1.0,
                verbose=true,
                tolerance=svm_tol,
                seed=i,
            )
            record_results!(df, settings, i, "L1R-SVM", l1rsvm_result, true_support, :cost)

            CSV.write(filename, df; append=true)
        end
    end
end

#
#   Program Execution
#
#   SEED FOR REPRODUCIBILITY: 1903
#
if length(ARGS) < 1
    error("Please provide a seed as an integer.")
end
seed = parse(Int, ARGS[1])
main(seed)
