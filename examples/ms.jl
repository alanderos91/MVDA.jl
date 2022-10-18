include("common.jl")

dir = ARGS[1]
@info "Output directory: $(dir)"

# Load the data
label_col = 6                           # column containing label
data_cols = 7:793                       # columns containing variables/predictors/features
df = CSV.read("/home/alanderos/Desktop/data/UCLA_CUI_2022-06-21.csv", DataFrame)

# Change labels for disease modifiers and class labels.
label_cols = ["active", "worsening", "progression", "new_mri_lesion", "class"]

# Settings
preshuffle = true
split = 0.5
nfolds = 5
nreplicates = 50
kernel = nothing
intercept = false
data_transform = NoTransformation

ne = 7
ng = 0
nl = 13
ns = 787
nhyper = (ne, ng, nl, ns)

# Drop a single class
for dropped_class in sort!(unique(df[!,6]), rev=true)
    tmp = filter(:NewClass => !isequal(dropped_class), df)
    example = string("drop", dropped_class)
    data = (tmp[!,label_col], Matrix{Float64}(tmp[!,data_cols]))
    run(dir, example, data, nhyper, preshuffle;
        at=split,           # CV set / Test set split
        nfolds=nfolds,      # number of folds
        data_transform=data_transform,
        nreplicates=nreplicates,
        kernel=kernel,
        intercept=intercept,
    )
end

# Use the entire dataset
example = "all"
data = (df[!,label_col], Matrix{Float64}(df[!,data_cols]))
run(dir, example, data, nhyper, preshuffle;
    at=split,           # CV set / Test set split
    nfolds=nfolds,      # number of folds
    data_transform=data_transform,
    nreplicates=nreplicates,
    kernel=kernel,
    intercept=intercept,
)
