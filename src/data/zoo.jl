function process_zoo(local_path, dataset)
    # Standardize format.
    dir = dirname(local_path)
    MVDA.process_dataset(local_path, dataset;
        label_mapping=string,
        header=false,
        class_index=18,
        variable_indices=2:17,
        ext=".csv",
    )    

    # Store column information.
    column_info = [
        "type",
        "hair",
        "feathers",
        "eggs",
        "milk",
        "airborne",
        "aquatic",
        "predator",
        "toothed",
        "backbone",
        "breathes",
        "venomous",
        "fins",
        "legs",
        "tail",
        "domestic",
        "catsize",
    ]

    info_file = joinpath(dir, "$(dataset).info")
    column_info_df = DataFrame(columns=column_info)
    CSV.write(info_file, column_info_df; writeheader=false, delim=',')

    return nothing
end

push!(
    MESSAGES[],
    """
    ## Dataset: zoo

    **7 classes / 101 instances / 16 variables**

    See: https://archive.ics.uci.edu/ml/datasets/zoo

    *Note*: This version strips 'animal name' from attributes.
    """
)

push!(REMOTE_PATHS[], "https://archive.ics.uci.edu/ml/machine-learning-databases/zoo/zoo.data")

push!(CHECKSUMS[], "cddc71c26ab9bc82795b8f4ff114cade41885d92720c6af29ffb69bcf73f0315")

push!(FETCH_METHODS[], DataDeps.fetch_default)

push!(POST_FETCH_METHODS[], path -> process_zoo(path, "zoo"))

push!(DATASETS[], "zoo")
