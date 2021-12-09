using CSV, DataFrames, MVDA, Latexify, Printf

function run(filenames, datasets)
    metrics = [:train, :validation, :test, :model]
    columns = Symbol[]
    for metric in metrics
        push!(columns, Symbol(metric, :_md))
        push!(columns, Symbol(metric, :_lo))
        push!(columns, Symbol(metric, :_hi))
    end
    columns = [:sparsity; columns]

    header = [
        "Sparsity",
        "Train",
        "Validation",
        "Test",
    ]

    output = DataFrame()
    for filename in filenames
        df_full = CSV.read(filename, DataFrame)
        df_reps = MVDA.cv_error(df_full)            # approximate CV error by averaging over folds
        df_ints = MVDA.credible_intervals(df_reps)  # create 95% equal-tailed credible intervals
        
        s_opt = df_ints.model_md[1]
        idx = searchsortedfirst(df_ints.sparsity, s_opt)
        row = df_ints[idx, columns]

        result = DataFrame(
            x1="$(round(row.model_md, digits=2)) ($(round(row.model_lo, digits=2)), $(round(row.model_hi, digits=2)))",
            x2="$(round(row.train_md, digits=2)) ($(round(row.train_lo, digits=2)), $(round(row.train_hi, digits=2)))",
            x3="$(round(row.validation_md, digits=2)) ($(round(row.validation_lo, digits=2)), $(round(row.validation_hi, digits=2)))",
            x4="$(round(row.test_md, digits=2)) ($(round(row.test_lo, digits=2)), $(round(row.test_hi, digits=2)))",
        )

        output = vcat(output, result)
    end

    return latexify(output,
        env=:table,
        booktabs=true,
        adjustment=:r,
        head=header,
        side=datasets,
        transpose=false,
        latex=false,
    )
end

# UCI
filenames = [
    "/home/alanderos/Desktop/VDA/iris-L-path=D2S.dat",
    "/home/alanderos/Desktop/VDA/lymphography-L-path=D2S.dat",
    "/home/alanderos/Desktop/VDA/zoo-L-path=D2S.dat",
    "/home/alanderos/Desktop/VDA/breast-cancer-wisconsin-L-path=D2S.dat",
    "/home/alanderos/Desktop/VDA/splice-L-path=D2S.dat",
    "/home/alanderos/Desktop/VDA/letter-recognition-L-path=D2S.dat",
    "/home/alanderos/Desktop/VDA/optdigits-L-path=D2S.dat",
    "/home/alanderos/Desktop/VDA/HAR-L-path=D2S.dat",
]

datasets = [
    "iris",
    "lymphography",
    "zoo",
    "bcw",
    "splice",
    "letters",
    "optdigits",
    "HAR",
]

display(run(filenames, datasets)); println()

# NONLINEAR
filenames = [
    "/home/alanderos/Desktop/VDA/clouds-NL-path=D2S.dat",
    "/home/alanderos/Desktop/VDA/circles-NL-path=D2S.dat",
    "/home/alanderos/Desktop/VDA/waveform-NL-path=D2S.dat",
    "/home/alanderos/Desktop/VDA/zoo-NL-path=D2S.dat",
    "/home/alanderos/Desktop/VDA/vowel-NL-path=D2S.dat",
    # "/home/alanderos/Desktop/VDA/HAR-NL-path=D2S.dat",
]

datasets = [
    "clouds",
    "circles",
    "waveform",
    "zoo",
    "vowel",
    # "HAR",
]

display(run(filenames, datasets)); println()

# CANCER
filenames = [
    "/home/alanderos/Desktop/VDA/leukemiaA-L-path=D2S.dat",
    "/home/alanderos/Desktop/VDA/prostate-L-path=D2S.dat",
    "/home/alanderos/Desktop/VDA/colon-L-path=D2S.dat",
    "/home/alanderos/Desktop/VDA/srbctA-L-path=D2S.dat",
    "/home/alanderos/Desktop/VDA/lymphomaA-L-path=D2S.dat",
    "/home/alanderos/Desktop/VDA/brain-L-path=D2S.dat",
]

datasets = [
    "leukemia",
    "prostate",
    "colon",
    "SRBCT",
    "lymphoma",
    "brain",
]

display(run(filenames, datasets)); println()
