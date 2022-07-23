function process_vowel(local_path, dataset)
    # Read both training and testing files.
    dir = dirname(local_path)
    tra = joinpath(dir, "vowel.train")
    tes = joinpath(dir, "vowel.test")
    df = vcat(
        CSV.read(tra, DataFrame, header=true),
        CSV.read(tes, DataFrame, header=true)
    )
    tmpfile = joinpath(dir, "$(dataset).tmp")
    CSV.write(tmpfile, df; writeheader=false, delim=',')

    # Standardize format.
    MVDA.process_dataset(tmpfile, dataset;
        label_mapping=string,
        header=false,
        class_index=2,
        variable_indices=3:ncol(df),
        ext=".csv",
    )
        
    # Store column information.
    info_file = joinpath(dir, "$(dataset).info")
    column_info = [["digit"]; ["$(i)x$(j)" for i in 1:8 for j in 1:8]]
    column_info_df = DataFrame(columns=column_info)
    CSV.write(info_file, column_info_df; writeheader=false, delim=',')

    # Clean up by removing separate training and testing files.
    rm(tra)
    rm(tes)

    return nothing
end

push!(
    MESSAGES[],
    """
    ## Dataset: vowel

    **11 classes / 990 instances / 10 variables**

    See: https://web.stanford.edu/~hastie/ElemStatLearn/ and https://web.stanford.edu/~hastie/ElemStatLearn/datasets/vowel.info.txt
    """
)

push!(
    REMOTE_PATHS[],
    [
        "https://web.stanford.edu/~hastie/ElemStatLearn/datasets/vowel.train",
        "https://web.stanford.edu/~hastie/ElemStatLearn/datasets/vowel.test"
    ],
)

push!(
    CHECKSUMS[],
    [
        "6285d72a3ec5af71286e6e0a778b836c229e3c41d6e0c2076294ee671e3ed9fb",
        "94720047d8e1d1f485b882347ad20b9b62fd04b080c855e23a10a4804e886f1b",
    ]
)

push!(
    FETCH_METHODS[],
    [
        DataDeps.fetch_default,
        DataDeps.fetch_default,
    ]
)

push!(
    POST_FETCH_METHODS[],
    [
        identity,
        path -> process_vowel(path, "vowel"),
    ]
)

push!(DATASETS[], "vowel")
