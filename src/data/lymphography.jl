function process_lymphography(local_path, dataset)
    function dummy_coding(x, nlevels)
        code = zeros(nlevels)
        code[x] = 1
        return code
    end
    convert_to_zero_one(x, ref=1) = x == ref ? 0 : 1

    function expand_variables!(data, row, i)
        data[i,1:4] = dummy_coding(row[2], 4)
        data[i,5] = convert_to_zero_one(row[3])
        data[i,6] = convert_to_zero_one(row[4])
        data[i,7] = convert_to_zero_one(row[5])
        data[i,8] = convert_to_zero_one(row[6])
        data[i,9] = convert_to_zero_one(row[7])
        data[i,10] = convert_to_zero_one(row[8])
        data[i,11] = convert_to_zero_one(row[9])
        data[i,12:14] = dummy_coding(row[10], 3)
        data[i,15:18] = dummy_coding(row[11], 4)
        data[i,19:21] = dummy_coding(row[12], 3)
        data[i,22:25] = dummy_coding(row[13], 4)
        data[i,26:29] = dummy_coding(row[14], 4)
        data[i,30:37] = dummy_coding(row[15], 8)
        data[i,38:40] = dummy_coding(row[16], 3)
        data[i,41] = convert_to_zero_one(row[17])
        data[i,42] = convert_to_zero_one(row[18])
        data[i,43:50] = dummy_coding(row[19], 8)
    end

    # Rewrite the original variables using dummy coding and o.
    tmpdf = CSV.read(local_path, DataFrame, header=false)
    data = Matrix{Int}(undef, nrow(tmpdf), 50)
    foreach(i -> expand_variables!(data, tmpdf[i,:], i), 1:nrow(tmpdf))
    df = DataFrame([tmpdf[:,1] data], :auto)

    dir = dirname(local_path)
    tmpfile = joinpath(dir, "$(dataset).tmp")
    CSV.write(tmpfile, df, writeheader=true)

    # Define a label mapping from integer to human-readable label.
    dict_wrapper(dict, label_i) = dict[label_i]
    class_dict = Dict(1 => "normal", 2 => "metastases", 3 => "malign lymph", 4 => "fibrosis")
    label_mapping = Base.Fix1(dict_wrapper, class_dict)

    # Standardize format.
    MVDA.process_dataset(tmpfile, dataset,
        label_mapping=label_mapping,
        header=true,
        class_index=1,
        variable_indices=2:42,
        ext=".csv"
    )

    # Set variable information.
    info_file = joinpath(dir, "$(dataset).info")
    column_info = [
        "class", 
        "normal", "lymphatics_arched", "lymphatics_deformed", "lymphatics_displaced", # normal (1) is ref
        "blk_of_affere",
        "blk_of_lymph_c",
        "blk_of_lymph_s",
        "bypass",
        "extravasates",
        "regeneration",
        "early_uptake",
        "lymph_nodes_dim_1", "lymph_nodes_dim_2", "lymph_nodes_dim_3", # 1 is ref
        "lymph_nodes_enl_2", "lymph_nodes_enl_3", "lymph_nodes_enl_4",
        "bean", "lymph_oval", "lymph_round", # bean is ref
        "defect_lacunar", "defect_lacunar_marginal", "defect_lacunar_central",
        "no_change", "node_lacunar", "node_lacunar_marginal", "node_lacunar_central", # no change is ref
        "no_change", "stru_grainy", "stru_droplike", "stru_coarse", "stru_diluted", "stru_reticular",
            "stru_stripped", "stru_faint", # no change is ref
        "no", "forms_chalices", "forms_vesicles", # no is ref 
        "dislocation",
        "exclusion",
        "n_nodes_0_9", "n_nodes_10_19", "n_nodes_20_29", "n_nodes_30_39", "n_nodes_40_49", "n_nodes_50_59",
            "n_nodes_60_69", "n_nodes_>=70", # 0-9 is ref
    ]
    column_info_df = DataFrame(columns=column_info)
    CSV.write(info_file, column_info_df; writeheader=false, delim=',')

    return nothing
end

push!(
    MESSAGES[],
    """
    ## Dataset: lymphography

    **4 classes / 148 instances / 50 variables**

    See: https://archive.ics.uci.edu/ml/datasets/Lymphography

    The original 18 categorical variables are coded as 50 binary variables.
    """
)

push!(REMOTE_PATHS[], "https://archive.ics.uci.edu/ml/machine-learning-databases/lymphography/lymphography.data")

push!(CHECKSUMS[], "e17d22d071341af08a505a10eb558afc835e302b02c8b381c1fa3dd305731935")

push!(FETCH_METHODS[], DataDeps.fetch_default)

push!(POST_FETCH_METHODS[], path -> process_lymphography(path, "lymphography"))

push!(DATASETS[], "lymphography")
