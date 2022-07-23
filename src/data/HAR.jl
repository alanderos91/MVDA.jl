function process_HAR(local_path, dataset)
    # Decompress the data. X data are in fixed width format.
    DataDeps.unpack(local_path, keep_originals=false)

    dir = dirname(local_path)
    tmpdir = joinpath(dir, "UCI HAR Dataset")
    train_dir = joinpath(tmpdir, "train")
    test_dir = joinpath(tmpdir, "test")
    data = vcat(
        CSV.read(joinpath(train_dir, "X_train.txt"), DataFrame, header=false, delim=' ', ignorerepeated=true),
        CSV.read(joinpath(test_dir, "X_test.txt"), DataFrame, header=false, delim=' ', ignorerepeated=true)
    )
    labels = vcat(
        CSV.read(joinpath(train_dir, "y_train.txt"), DataFrame, header=false),
        CSV.read(joinpath(test_dir, "y_test.txt"), DataFrame, header=false)
    )
    df = hcat(labels, data, makeunique=true)

    # Write to a temporary file.
    tmpfile = joinpath(dir, "$(dataset).tmp")
    CSV.write(tmpfile, df, writeheader=true)

    # Define a label mapping.
    dict_wrapper(dict, label_i) = dict[label_i]
    activity_labels_df = CSV.read(joinpath(tmpdir, "activity_labels.txt"), DataFrame, header=false, delim=' ', ignorerepeated=true)
    activity_labels_dict = Dict(row[1] => row[2] for row in eachrow(activity_labels_df))
    label_mapping = Base.Fix1(dict_wrapper, activity_labels_dict)

    # Standardize format.
    MVDA.process_dataset(tmpfile, dataset,
        label_mapping=label_mapping,
        header=true,
        class_index=1,
        variable_indices=2:ncol(df),
        ext=".csv.gz"
    )

    # Extract variable information.
    features_df = CSV.read(joinpath(tmpdir, "features.txt"), DataFrame, header=false, delim=' ', ignorerepeated=true)
    info_file = joinpath(dir, "$(dataset).info")
    column_info = [["activity"]; features_df[:,2]]
    column_info_df = DataFrame(columns=column_info)
    CSV.write(info_file, column_info_df; writeheader=false, delim=',')

    # Clean up.
    rm("__MACOSX", recursive=true)
    rm(tmpdir, recursive=true)

    return nothing
end

push!(
    MESSAGES[],
    """
    ## Dataset: HAR

    **6 classes / 10299 instances / 561 variables**

    See: https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones

    Check the README.txt file for further details about this dataset.

    **Notes**: 

    - Features are normalized and bounded within [-1,1].
    - Each feature vector is a row on the text file.
    - The units used for the accelerations (total and body) are 'g's (gravity of earth -> 9.80665 m/seg2).
    - The gyroscope units are rad/seg.
    - A video of the experiment including an example of the 6 recorded activities with one of the participants can be seen in the following link: http://www.youtube.com/watch?v=XOEN9W05_4A

    For more information about this dataset please contact: activityrecognition '@' smartlab.ws

    **License**:

    Use of this dataset in publications must be acknowledged by referencing the following publication [1] 

    [1] Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra and Jorge L. Reyes-Ortiz. A Public Domain Dataset for Human Activity Recognition Using Smartphones. 21th European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning, ESANN 2013. Bruges, Belgium 24-26 April 2013. 

    This dataset is distributed AS-IS and no responsibility implied or explicit can be addressed to the authors or their institutions for its use or misuse. Any commercial use is prohibited.
    """
)

push!(REMOTE_PATHS[], "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip")

push!(CHECKSUMS[], "2045e435c955214b38145fb5fa00776c72814f01b203fec405152dac7d5bfeb0")

push!(FETCH_METHODS[], DataDeps.fetch_default)

push!(POST_FETCH_METHODS[], path -> process_HAR(path, "HAR"))

push!(DATASETS[], "HAR")
