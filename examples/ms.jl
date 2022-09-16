include("common.jl")

dir = ARGS[1]
@info "Output directory: $(dir)"

# Load the data
label_col = 6                           # column containing label
data_cols = 7:793                       # columns containing variables/predictors/features
df = CSV.read("/home/alanderos/Desktop/data/UCLA_CUI_2022-06-21.csv", DataFrame)

# Settings
preshuffle = true
split = 0.5
nfolds = 3
nreplicates = 50
kernel = nothing
intercept = false
data_transform = NoTransformation

ne = 11
ng = 0
nl = 100
ns = 767
nhyper = (ne, ng, nl, ns)

# Use the entire dataset
example = "all"
data = (Vector(df[!,label_col]), Matrix{Float64}(df[!,data_cols]))
run(dir, example, data, nhyper, preshuffle;
    at=split,           # CV set / Test set split
    nfolds=nfolds,      # number of folds
    data_transform=data_transform,
    nreplicates=nreplicates,
    kernel=kernel,
    intercept=intercept,
)

# Drop items where sublabels contain missing items
example = "dropmissing"
tmp = dropmissing(df)
data = (Vector(tmp[!,label_col]), Matrix{Float64}(tmp[!,data_cols]))
run(dir, example, data, nhyper, preshuffle;
    at=split,           # CV set / Test set split
    nfolds=nfolds,      # number of folds
    data_transform=data_transform,
    nreplicates=nreplicates,
    kernel=kernel,
    intercept=intercept,
)

# Combine class B and class C
example = "combined"
tmp = copy(df)
for i in eachindex(tmp.NewClass)
    if tmp.NewClass[i] == "C"
        tmp.NewClass[i] = "B"
    end
end
data = (Vector(tmp[!,label_col]), Matrix{Float64}(tmp[!,data_cols]))
run(dir, example, data, nhyper, preshuffle;
    at=split,           # CV set / Test set split
    nfolds=nfolds,      # number of folds
    data_transform=data_transform,
    nreplicates=nreplicates,
    kernel=kernel,
    intercept=intercept,
)

# Drop features with a single observation
example = "features"
data = (Vector(df[!,label_col]), Matrix{Float64}(df[!,data_cols]))
idx = findall(x -> sum(x) > 1, eachcol(data[2]))
data = (data[1], data[2][:,idx])
run(dir, example, data, nhyper, preshuffle;
    at=split,           # CV set / Test set split
    nfolds=nfolds,      # number of folds
    data_transform=data_transform,
    nreplicates=nreplicates,
    kernel=kernel,
    intercept=intercept,
)
