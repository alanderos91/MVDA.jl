include("common.jl")

# Gaussian Clouds
function clouds()
    n_cv, n_test = 250, 10^3
    nsamples = n_cv + n_test
    nclasses = 3
    L, X = MVDA.gaussian_clouds(nsamples, nclasses; sigma=0.25, rng=StableRNG(1903))
    DataFrame([L X], :auto)
end

# Nested Circles
function circles()
    n_cv, n_test = 250, 10^3
    nsamples = n_cv + n_test
    nclasses = 3
    L, X = MVDA.nested_circles(nsamples, nclasses; p=8//10, rng=StableRNG(1903))
    DataFrame([L X], :auto)
end

# Waveform
function waveform()
    n_cv, n_test = 250, 10^3
    nsamples = n_cv + n_test
    nfeatures = 21
    L, X = MVDA.waveform(nsamples, nfeatures; rng=StableRNG(1903))
    DataFrame([L X], :auto)
end

function run_all_examples(dir, examples, projections)
    for (example, prob_settings, cv_settings, nhyper, data_transform, preshuffle) in examples
        df = if example == "clouds"
            clouds()
        elseif example == "circles"
            circles()
        elseif example == "waveform"
            waveform()
        else
            MVDA.dataset(example)
        end
        data = (string.(df[!,1]), Matrix{Float64}(df[!,2:end]))
        kernel, intercept = prob_settings
        (nfolds, nreplicates, split) = cv_settings
    
        for projection_type in projections
            run(dir, example, data, nhyper, projection_type, preshuffle;
                at=split,           # CV set / Test set split
                nfolds=nfolds,      # number of folds
                data_transform=data_transform,
                nreplicates=nreplicates,
                kernel=kernel,
                intercept=intercept,
            )
        end
    end
end

#
# Begin script
#

dir = ARGS[1]
@info "Output directory: $(dir)"

examples = (
    # format:
    #
    #   1. example name
    #   2. number of folds
    #   3. number of replicates
    #   4. proportion for cross validation set
    #   5. tuple: number epsilon values, number gamma values, number lambda values, number sparsity values
    #   6. data transformation for normalization/standardization
    #
    ("clouds", (RBFKernel(), true,), (5, 50, 250/1250,), (1, 11, 97, 200), ZScoreTransform, true),
    ("circles", (RBFKernel(), true,), (5, 50, 250/1250,), (1, 5, 97, 200), ZScoreTransform, true),
    ("waveform", (RBFKernel(), true,), (5, 50, 250/1250,), (1, 5, 97, 200), ZScoreTransform, true),
    ("spiral", (RBFKernel(), true,), (5, 50, 0.5,), (5, 5, 97, 200), ZScoreTransform, true),
    ("spiral-hard", (RBFKernel(), true,), (5, 50, 0.5,), (1, 5, 97, 200), ZScoreTransform, true),
    # use cv / test split in original dataset: 528 + 462
    ("vowel", (RBFKernel(), true,), (5, 50, 528 / 990,), (1, 5, 97, 100), NoTransformation, false),
    # # use cv / test split in original dataset: 7352 + 2947
    # ("HAR", (RBFKernel(), true,), (5, 10, 7352 / 10299,), (1, 5, 25, 200), NoTransformation, false),
)

projections = (
    HomogeneousL0Projection, HeterogeneousL0Projection,
    HomogeneousL1BallProjection, HeterogeneousL1BallProjection
)

#
#   MM
#

run_all_examples(dir, examples, projections)
