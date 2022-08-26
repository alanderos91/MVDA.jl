include("common.jl")

# Examples ordered from easiest to hardest
examples = (
    # format:
    #
    #   1. example name
    #   2. number of folds
    #   3. number of replicates
    #   4. proportion for cross validation set
    #   5. tuple: number epsilon values, number lambda values, number sparsity values
    #   6. data transformation for normalization/standardization
    #
    ("colon", (nothing, true,), (3, 50, 0.8,), (11,0,13,50), ZScoreTransform),
    ("srbctA", (nothing, true,), (3, 50, 0.8,), (11,0,13,50), ZScoreTransform),
    ("leukemiaA", (nothing, true,), (3, 50, 0.8,), (11,0,13,50), ZScoreTransform),
    ("lymphomaA", (nothing, true,), (3, 50, 0.8,), (11,0,13,50), ZScoreTransform),
    ("brain", (nothing, true,), (3, 50, 0.8,), (11,0,13,50), ZScoreTransform),
    ("prostate", (nothing, true,), (3, 50, 0.8,), (11,0,13,50), ZScoreTransform),
)

dir = ARGS[1]
@info "Output directory: $(dir)"

for (example, prob_settings, cv_settings, nhyper, data_transform) in examples
    df = CSV.read("/home/alanderos/Desktop/data/$(example).DAT", DataFrame, header=false, delim=' ',)
    for j in 1:ncol(df)-1
        if eltype(df[!,j]) <: AbstractString
            df[!,j] .= parse.(Float64, df[!,j])
        end
    end
    data = (string.(df[!,end]), Matrix{Float64}(df[!,1:end-1]))
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
