using MLDataUtils
include("cv-driver.jl") # load the rest

N = 50
file = "/home/alanderos/Desktop/VDA/cancer-timings.txt"
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
        nfolds=3,    # number of folds
        gtol=1e-6,   # tolerance on gradient for convergence of inner problem
        dtol=1e-6,   # tolerance on distance for convergence of outer problem
        rtol=0.0,    # use strict distance criteria
        nesterov_threshold=100, # delay on Nesterov acceleration
    )
end

for dataset in ("colon", "srbctA", "leukemiaA", "lymphomaA", "brain", "prostate",)
    df = CSV.read("/home/alanderos/Desktop/data/$(dataset).DAT", DataFrame, header=false)
    data = (Vector(df[!,end]), Matrix{Float64}(df[!,1:end-1]))
    ((n, p), c) = size(data[2]), length(unique(data[1]))
    standardize!(data[2])
    idx = randperm(MersenneTwister(16), n)
    _permute!(data[1], idx)
    _permutecols!(data[2], idx)
    s_grid = [1-k/p for k in p:-1:0]

    (cv_set, test_set) = splitobs(data, at=0.8, obsdim=1)
    filename = dataset
    title = "$(dataset) / $(n) samples / $(p) features / $(c) classes"
    t1, s1, TR1, V1, T1 = run("$(filename)-path=D2S", title, cv_set, test_set, s_grid, false)   # dense -> sparse
    t2, s2, TR2, V2, T2 = run("$(filename)-path=S2D", title, cv_set, test_set, s_grid, true)    # sparse -> dense

    open(file, "a") do io
        println(io, N,",",dataset,",",false,",",t1,",",join(s1,','),",",join(TR1,','),",",join(V1,','),",",join(T1,','))
        println(io, N,",",dataset,",",true,",",t2,",",join(s2,','),",",join(TR2,','),",",join(V2,','),",",join(T2,','))
    end
end
