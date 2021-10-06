using DataDeps

register(DataDep(
    "vowel",
    """
    Dataset: vowel
    Website: https://web.stanford.edu/~hastie/ElemStatLearn/

    Observations: 990 (528 + 462)
    Features:     10
    Classes:      11

    This info is the original source information for these data.

        NAME: Vowel Recognition (Deterding data)
        
        SUMMARY: Speaker independent recognition of the eleven steady state vowels
        of British English using a specified training set of lpc derived log area
        ratios.
        
        SOURCE: David Deterding  (data and non-connectionist analysis)
                Mahesan Niranjan (first connectionist analysis)
                Tony Robinson    (description, program, data, and results)
        
        To contact Tony Robinson by electronic mail, use address 
        "ajr@dsl.eng.cam.ac.uk"
        
        MAINTAINER: neural-bench@cs.cmu.edu

    See the complete information at https://web.stanford.edu/~hastie/ElemStatLearn/datasets/vowel.info.txt
    """,
    [
    "https://web.stanford.edu/~hastie/ElemStatLearn/datasets/vowel.train",
    "https://web.stanford.edu/~hastie/ElemStatLearn/datasets/vowel.test"
    ],
    "f6f7d76de6247e85add6ec3e0d5988f7406338f1562897e558844ae750b6b6e0",
    #
    # Disclaimer: Want to combine both .train and .test into a single .csv.
    # Design of post_fetch_method may not have been designed with this in mind.
    # Workaround here is to do nothing to first file, then assume first file has been
    # downloaded and can be loaded in the second function.
    # This works in practice, but will fail when running preupload_check.
    # 
    # Reason: `path` is randomly generated and differs between the two calls.
    #
    post_fetch_method = [
    identity,
    path -> begin
        # Read both training and testing files.
        dir = dirname(path)
        tra = joinpath(dir, "vowel.train")
        tes = joinpath(dir, "vowel.test")
        df = vcat(
            CSV.read(tra, DataFrame, header=true),
            CSV.read(tes, DataFrame, header=true)
        )
        
        # Process the data as usual.
        MVDA.process_dataset(df,
            target_index=2,
            feature_indices=3:ncol(df),
            ext=".csv",
        )

        # Clean up by removing separate training and testing files.
        rm(tra)
        rm(tes)
    end],
))
