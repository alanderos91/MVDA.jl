include("common.jl")

function run_all_examples(dir, examples, projections)
    for (example, prob_settings, cv_settings, nhyper, data_transform, preshuffle) in examples
        df = CSV.read("/home/alanderos/Desktop/data/$(example).DAT", DataFrame, header=false, delim=' ',)
        for j in 1:ncol(df)-1
            if eltype(df[!,j]) <: AbstractString
                df[!,j] .= parse.(Float64, df[!,j])
            end
        end
        data = (string.(df[!,end]), Matrix{Float64}(df[!,1:end-1]))
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
    ("colon", (nothing, true,), (3, 50, 0.8,), (11,0,97,1000), ZScoreTransform, true),
    ("srbctA", (nothing, true,), (3, 50, 0.8,), (11,0,97,1000), ZScoreTransform, true),
    ("leukemiaA", (nothing, true,), (3, 50, 0.8,), (11,0,97,1750), ZScoreTransform, true),
    ("lymphomaA", (nothing, true,), (3, 50, 0.8,), (11,0,97,2000), ZScoreTransform, true),
    ("brain", (nothing, true,), (3, 50, 0.8,), (11,0,97,2750), ZScoreTransform, true),
    ("prostate", (nothing, true,), (3, 50, 0.8,), (11,0,97,3000), ZScoreTransform, true),
)

projections = (
    HomogeneousL0Projection, HeterogeneousL0Projection,
    HomogeneousL1BallProjection, HeterogeneousL1BallProjection
)

#
#   MM
#

run_all_examples(dir, examples, projections)
