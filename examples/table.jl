using CSV, DataFrames, MVDA, Latexify, Printf

function run(files)
    metrics = [:train, :validation, :test, :model]
    columns = Symbol[]
    for metric in metrics
        push!(columns, Symbol(metric, :_md))
        push!(columns, Symbol(metric, :_lo))
        push!(columns, Symbol(metric, :_hi))
    end
    columns = [:sparsity; columns]

    output = DataFrame()
    datasets = String[]
    for file in files
        # strip the dataset name
        push!(datasets, first(split(basename(file), '-')))

        df_full = CSV.read(file, DataFrame)
        df_reps = MVDA.cv_error(df_full)            # approximate CV error by averaging over folds
        df_ints = MVDA.credible_intervals(df_reps)  # create 95% equal-tailed credible intervals
        
        s_opt = df_ints.model_md[1]
        idx = searchsortedfirst(df_ints.sparsity, s_opt)
        row = df_ints[idx, columns]

        result = DataFrame(
            x1="$(round(row.model_md, digits=2))",
            x2="$(round(row.model_lo, digits=2))",
            x3="$(round(row.model_hi, digits=2))",
            x4="$(round(row.train_md, digits=2))",
            x5="$(round(row.train_lo, digits=2))",
            x6="$(round(row.train_hi, digits=2))",
            x7="$(round(row.validation_md, digits=2))",
            x8="$(round(row.validation_lo, digits=2))",
            x9="$(round(row.validation_hi, digits=2))",
            x10="$(round(row.test_md, digits=2))",
            x11="$(round(row.test_lo, digits=2))",
            x12="$(round(row.test_hi, digits=2))",
            x13=first(df_full.nfeatures)
        )

        output = vcat(output, result)
    end

    # sort by features and drop the last column
    idx = sortperm(output, [:x13])
    output = output[idx,1:end-1]
    datasets .= datasets[idx]

    return latexify(output,
        env=:table,
        booktabs=true,
        adjustment=:r,
        side=datasets,
        transpose=false,
        latex=false,
    )
end

dir = ARGS[1]
@info "Reading from directory: $(dir)"
files = readdir(dir, join=true)
filter!(contains(".dat"), files)
display(run(files))
