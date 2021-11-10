using MLDataUtils
include("cv-driver.jl") # load the rest

N = 50
file = "/home/alanderos/Desktop/VDA/uci-timings.txt"
isfile(file) && rm(file)
open(file, "a") do io
    s_str = subheader("s")
    TR_str = subheader("train")
    V_str = subheader("validation")
    T_str = subheader("test")
    println(io, "replicates,dataset,sparse2dense,time,$(s_str),$(TR_str),$(V_str),$(T_str)")
end

run = function(fname, title, cv_set, test_set, s_grid, sparse2dense)
    run_example(MersenneTwister(123456), fname, title, cv_set, test_set, N, s_grid, sparse2dense;
        nouter=10^2, # outer iterations
        ninner=10^5, # inner iterations
        nfolds=5,    # number of folds
        gtol=1e-6,   # tolerance on gradient for convergence of inner problem
        dtol=1e-6,   # tolerance on distance for convergence of outer problem
        rtol=0.0,    # use strict distance criteria
        nesterov_threshold=100, # delay on Nesterov acceleration
    )
end

for example in ("iris", "lymphography", "zoo", "breast-cancer-wisconsin", "splice", "letter-recognition", "optdigits", "HAR")
    # load the example
    df = MVDA.dataset(example)
    data = (Vector(df[!,1]), Matrix{Float64}(df[!,2:end]))

    # purge features without variation by checking whether minimum(x) == maximum(x)
    idx = findall(v -> v[1] != v[2], map(extrema, eachcol(data[2])))
    data = (data[1], data[2][:, idx])

    # extract problem info, shuffle, and create grid
    ((n, p), c) = size(data[2]), length(unique(data[1]))
    standardize!(data[2])
    idx = randperm(MersenneTwister(16), n)
    _permute!(data[1], idx)
    _permutecols!(data[2], idx)
    s_grid = [1-k/p for k in p:-1:0]

    # split data for CV / testing and fit a model
    (cv_set, test_set) = splitobs(data, at=0.8, obsdim=1)
    filename = example
    title = "$(example) / $(n) samples / $(p) features / $(c) classes"
    t, s, TR, V, T = run("$(filename)-path=D2S", title, cv_set, test_set, s_grid, false) # dense -> sparse

    open(file, "a") do io
        println(io, N,",",example,",",false,",",t,",",join(s,','),",",join(TR,','),",",join(V,','),",",join(T,','))
    end
end
