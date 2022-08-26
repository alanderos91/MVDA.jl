include("common.jl")

# Waveform example
waveform_data = begin
    n_cv, n_test = 375, 10^3
    nsamples = n_cv + n_test
    nfeatures = 21
    Y, X = MVDA.waveform(nsamples, nfeatures; rng=StableRNG(1903))
    DataFrame([Y X], :auto)
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
    ("iris", (nothing, true,), (3, 50, 120/150,), (11, 0, 13, 4), ZScoreTransform),
    ("lymphography", (nothing, false,), (3, 50, 105 / 148,), (11, 0, 13, 100), NoTransformation),
    ("synthetic", (nothing, false,), (5, 50, 0.8,), (11, 0, 13, 100), ZScoreTransform),
    ("synthetic-hard", (nothing, false,), (5, 50, 0.8,), (11, 0, 13, 100), ZScoreTransform),
    ("zoo", (nothing, true,), (3, 50, 0.9,), (11, 0, 13, 50), NoTransformation),
    ("bcw", (nothing, true,), (5, 50, 0.8,), (11, 0, 13, 50), NoTransformation),
    ("waveform", (nothing, false,), (5, 50, 375 / 1375,), (11, 0, 13, 50), ZScoreTransform),
    ("splice", (nothing, false,), (5, 50, 0.8,), (11, 0, 13, 50), NoTransformation),
    ("letters", (nothing, true,), (5, 50, 0.8,), (11, 0, 13, 50), NoTransformation),
    ("optdigits", (nothing, true,), (5, 50, 3823 / 5620,), (11, 0, 13, 64), NoTransformation),  # use cv / test split in original dataset: 3823 + 1797
    ("vowel", (nothing, true,), (5, 50, 528 / 990,), (11, 0, 13, 50), NoTransformation),    # use cv / test split in original dataset: 528 + 462
    ("TCGA-HiSeq", (nothing, true,), (4, 10, 0.75,), (11, 0, 13, 250), ZScoreTransform),
    ("HAR", (nothing, true,), (5, 10, 7352 / 10299,), (11, 0, 13, 50), NoTransformation),   # use cv / test split in original dataset: 7352 + 2947
)

dir = ARGS[1]
@info "Output directory: $(dir)"

for (example, prob_settings, cv_settings, nhyper, data_transform) in examples
    df = example != "waveform" ? MVDA.dataset(example) : waveform_data
    data = (string.(df[!,1]), Matrix{Float64}(df[!,2:end]))
    kernel, intercept = prob_settings
    (nfolds, nreplicates, split) = cv_settings
    run(dir, example, data, nhyper, false;
        at=split,           # CV set / Test set split
        nfolds=nfolds,      # number of folds
        data_transform=data_transform,
        nreplicates=nreplicates,
        kernel=kernel,
        intercept=intercept,
    )
end
