using DataDeps

register(DataDep(
    "zoo",
    """
    Dataset: zoo
    Author: Richard Forsyth
    Website: https://archive.ics.uci.edu/ml/datasets/zoo

    Observations: 101
    Features:     17 (16 numeric)
    Classes:      7

    Note: This version strips 'animal name' from attributes.
    """,
    "https://archive.ics.uci.edu/ml/machine-learning-databases/zoo/zoo.data",
    "cddc71c26ab9bc82795b8f4ff114cade41885d92720c6af29ffb69bcf73f0315",
    post_fetch_method = (path -> begin
        MVDA.process_dataset(path,
            header=false,
            target_index=18,
            feature_indices=2:17,
            ext=".csv",
        )
    end),
))
