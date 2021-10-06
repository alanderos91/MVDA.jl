using DataDeps

register(DataDep(
    "lymphography",
    """
    Dataset: lymphography
    Author: Igor Kononenko, Bojan Cestnik
    Website: https://archive.ics.uci.edu/ml/datasets/Lymphography

    Observations: 148
    Features:     18
    Classes:      4

    This is one of three domains provided by the Oncology Institute that has repeatedly appeared in the machine learning literature. (See also breast-cancer and primary-tumor.)
    """,
    "https://archive.ics.uci.edu/ml/machine-learning-databases/lymphography/lymphography.data",
    "e17d22d071341af08a505a10eb558afc835e302b02c8b381c1fa3dd305731935",
    post_fetch_method = (path -> begin
        MVDA.process_dataset(path,
            header=false,
            target_index=1,
            feature_indices=2:19,
            ext=".csv",
        )
    end),
))
