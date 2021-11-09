using MLDataUtils, KernelFunctions
include("cv-driver.jl") # load the rest

N = 50
file = "/home/alanderos/Desktop/VDA/nonlinear-timings.txt"
isfile(file) && rm(file)
open(file, "a") do io
    s_str = subheader("s")
    TR_str = subheader("train")
    V_str = subheader("validation")
    T_str = subheader("test")
    println(io, "replicates,dataset,sparse2dense,time,$(s_str),$(TR_str),$(V_str),$(T_str)")
end

run = function(fname, title, data, sparse2dense)
    # extract problem info, shuffle, and create grid
    ((n, _), _) = size(data[2]), length(unique(data[1]))
    standardize!(data[2])
    idx = randperm(MersenneTwister(16), n)
    _permute!(data[1], idx)
    _permutecols!(data[2], idx)
    s_grid = [1-k/n for k in n:-1:0]
    dist = zeros(binomial(n, 2))
    K = 1
    for i in 1:n, j in i+1:n
        if data[1][i] != data[1][j]
            dist[K] = norm(data[2][i,:] - data[2][j,:])
            K += 1
        end
    end
    σ = median(dist[1:K])
    kernel = σ * RBFKernel()

    (cv_set, test_set) = splitobs(data, at=0.8, obsdim=1)
    run_nonlinear_example(MersenneTwister(123456),
        fname, title, kernel, cv_set, test_set, N, s_grid, sparse2dense;
        nouter=10^2, # outer iterations
        ninner=10^6, # inner iterations
        nfolds=3,    # number of folds
        gtol=1e-6,   # tolerance on gradient for convergence of inner problem
        dtol=1e-6,   # tolerance on distance for convergence of outer problem
        rtol=0.0,    # use strict distance criteria
        nesterov_threshold=100, # delay on Nesterov acceleration
    )
end

# Nested Circle
n = 200
c = 3
data = MVDA.generate_nested_circle(n, c; p=8//10, rng=MersenneTwister(1903))
p = size(data[2], 2)
example = "circles"
title = "$(example) / $(n) samples / $(p) features / $(c) classes"
t, s, TR, V, T = run("$(example)-NL-path=D2S", title, data, false)
open(file, "a") do io
    println(io, N,",",example,",",false,",",t,",",join(s,','),",",join(TR,','),",",join(V,','),",",join(T,','))
end

# Waveform
n = 300
p = 21
data = MVDA.generate_waveform(n, p; rng=MersenneTwister(1903))
c = length(unique(data[1]))
example = "waveform"
title = "$(example) / $(n) samples / $(p) features / $(c) classes"
t, s, TR, V, T = run("$(example)-NL-path=D2S", title, data, false)
open(file, "a") do io
    println(io, N,",",example,",",false,",",t,",",join(s,','),",",join(TR,','),",",join(V,','),",",join(T,','))
end

# Zoo
df = MVDA.dataset("zoo")
data = (Vector(df[!,1]), Matrix{Float64}(df[!,2:end]))
((n, p), c) = size(data[2]), length(unique(data[1]))
example = "zoo"
title = "$(example) / $(n) samples / $(p) features / $(c) classes"
t, s, TR, V, T = run("$(example)-NL-path=D2S", title, data, false)
open(file, "a") do io
    println(io, N,",",example,",",false,",",t,",",join(s,','),",",join(TR,','),",",join(V,','),",",join(T,','))
end

# Vowel
df = MVDA.dataset("vowel")
data = (Vector(df[!,1]), Matrix{Float64}(df[!,2:end]))
((n, p), c) = size(data[2]), length(unique(data[1]))
example = "vowel"
title = "$(example) / $(n) samples / $(p) features / $(c) classes"
t, s, TR, V, T = run("$(example)-NL-path=D2S", title, data, false)
open(file, "a") do io
    println(io, N,",",example,",",false,",",t,",",join(s,','),",",join(TR,','),",",join(V,','),",",join(T,','))
end
