using DataDeps

register(DataDep(
    "HAR",
    """
    Dataset: HAR
    Website: https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones

    Observations: 10299 (7352 + 2947)
    Features:     561
    Classes:      6

    Jorge L. Reyes-Ortiz(1,2), Davide Anguita(1), Alessandro Ghio(1), Luca Oneto(1) and Xavier Parra(2)
    
        1 - Smartlab - Non-Linear Complex Systems Laboratory
        DITEN - Università degli Studi di Genova, Genoa (I-16145), Italy.
    
        2 - CETpD - Technical Research Centre for Dependency Care and Autonomous Living
        Universitat Politècnica de Catalunya (BarcelonaTech). Vilanova i la Geltrú (08800), Spain

    activityrecognition '@' smartlab.ws

    The experiments have been carried out with a group of 30 volunteers within an age bracket of 19-48 years. Each person performed six activities (WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING) wearing a smartphone (Samsung Galaxy S II) on the waist. Using its embedded accelerometer and gyroscope, we captured 3-axial linear acceleration and 3-axial angular velocity at a constant rate of 50Hz. The experiments have been video-recorded to label the data manually. The obtained dataset has been randomly partitioned into two sets, where 70% of the volunteers was selected for generating the training data and 30% the test data.

    The sensor signals (accelerometer and gyroscope) were pre-processed by applying noise filters and then sampled in fixed-width sliding windows of 2.56 sec and 50% overlap (128 readings/window). The sensor acceleration signal, which has gravitational and body motion components, was separated using a Butterworth low-pass filter into body acceleration and gravity. The gravitational force is assumed to have only low frequency components, therefore a filter with 0.3 Hz cutoff frequency was used. From each window, a vector of features was obtained by calculating variables from the time and frequency domain.

    Check the README.txt file for further details about this dataset.

    A video of the experiment including an example of the 6 recorded activities with one of the participants can be seen in the following link: http://www.youtube.com/watch?v=XOEN9W05_4A

    An updated version of this dataset can be found at
    
    http://archive.ics.uci.edu/ml/datasets/Smartphone-Based+Recognition+of+Human+Activities+and+Postural+Transitions.
    
    It includes labels of postural transitions between activities and also the full raw inertial signals instead of the ones pre-processed into windows. 
    """,
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip",
    "2045e435c955214b38145fb5fa00776c72814f01b203fec405152dac7d5bfeb0",
    # customize post fetch method to combine files and save in compressed format
    post_fetch_method = (path -> begin
        DataDeps.unpack(path, keep_originals=false)

        # Read the data and labels files. X data are in fixed width format.
        tmpdir = "UCI HAR Dataset"
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
        tmpdf = hcat(labels, data, makeunique=true)

        # Process the file as usual.
        MVDA.process_dataset(tmpdf,
            target_index=1,
            feature_indices=2:ncol(tmpdf),
            ext=".csv.gz"
        )

        # Clean up.
        rm("__MACOSX", recursive=true)
        rm(tmpdir, recursive=true)
    end),
))
