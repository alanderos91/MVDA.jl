include("common.jl")

# Waveform example
function waveform()
    n_cv, n_test = 375, 10^3
    nsamples = n_cv + n_test
    nfeatures = 21
    L, X = MVDA.waveform(nsamples, nfeatures; rng=StableRNG(1903))
    DataFrame([L X], :auto)
end

examples = (
    # format:
    #
    #   1. string: example name
    #   2. tuple: kernel choice (nothing => linear) and intercept indicator
    #   3. tuple: (number of folds, number of replicates, proportion in CV set)
    #   4. tuple: number of (epsilon, gamma, lambda, sparsity) values
    #   5. type: data transformation for normalization/standardization
    #
    ("iris", (nothing, true,), (3, 100, 120/150,), (11, 0, 45, 4), ZScoreTransform, true),
    ("lymphography", (nothing, true,), (3, 100, 105 / 148,), (11, 0, 45, 100), NoTransformation, true),
    ("synthetic", (nothing, false,), (5, 50, 0.8,), (11, 0, 17, 100), ZScoreTransform, true),
    ("synthetic-hard", (nothing, false,), (5, 50, 0.8,), (11, 0, 17, 100), ZScoreTransform, true),
    ("zoo", (nothing, true,), (3, 50, 0.9,), (11, 0, 25, 50), NoTransformation, true),
    ("bcw", (nothing, true,), (5, 50, 0.8,), (11, 0, 25, 50), NoTransformation, true),
    ("waveform", (nothing, false,), (5, 50, 375 / 2575,), (11, 0, 25, 50), ZScoreTransform, true),
    ("splice", (nothing, true,), (5, 50, 0.685,), (3, 0, 13, 240), NoTransformation, true),
    ("letters", (nothing, true,), (5, 50, 0.8,), (11, 0, 17, 50), NoTransformation, true),
    # use cv / test split in original dataset: 3823 + 1797
    ("optdigits", (nothing, true,), (5, 50, 3823 / 5620,), (11, 0, 17, 64), NoTransformation, false),
    # use cv / test split in original dataset: 528 + 462
    ("vowel", (nothing, true,), (5, 50, 528 / 990,), (11, 0, 45, 50), NoTransformation, false),
    # use cv / test split in original dataset: 7352 + 2947
    ("HAR", (nothing, true,), (5, 10, 7352 / 10299,), (1, 0, 5, 250), NoTransformation, false),
    ("TCGA-HiSeq", (nothing, true,), (4, 10, 0.75,), (1, 0, 5, 10000), ZScoreTransform, true),
)

dir = ARGS[1]
@info "Output directory: $(dir)"

for (example, prob_settings, cv_settings, nhyper, data_transform, preshuffle) in examples
    df = if example == "waveform"
        waveform()
    else
        MVDA.dataset(example)
    end
    data = (string.(df[!,1]), Matrix{Float64}(df[!,2:end]))
    kernel, intercept = prob_settings
    (nfolds, nreplicates, split) = cv_settings
    run(dir, example, data, nhyper, preshuffle;
        at=split,           # CV set / Test set split
        nfolds=nfolds,      # number of folds
        data_transform=data_transform,
        nreplicates=nreplicates,
        kernel=kernel,
        intercept=intercept,
    )
end
