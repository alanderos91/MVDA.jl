using MLDataUtils
include("cv-driver.jl") # load the rest

N = 100
c = 4
p = 100
ϵ = 0.5 * sqrt(2*c/(c-1))
d_vals = (1.0, 2.0, 3.0)
s_grid = [1-k/p for k in p:-1:0]

file = "/home/alanderos/Desktop/VDA/WS2007-timings.txt"
isfile(file) && rm(file)
open(file, "a") do io
    println(io, "replicates,n,p,c,d,sparse2dense,time")
end

run = function(fname, title, cv_set, test_set, sparse2dense)
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

# underdetermined
nsamples = 20
n = nsamples*c
for d in d_vals # go hard -> easy
    data = MVDA.simulate_WS2007(n, p, c, nsamples, d)
    standardize!(data[2])
    idx = randperm(MersenneTwister(16), n)
    _permute!(data[1], idx)
    _permutecols!(data[2], idx)
    
    (cv_set, test_set) = splitobs(data, at=0.8, obsdim=1)
    filename = "underdetermined-d=$(d)"
    title = "d = $(d)"
    t1 = run("$(filename)-path=D2S", title, cv_set, test_set, false) # dense -> sparse
    t2 = run("$(filename)-path=S2D", title, cv_set, test_set, true)  # sparse -> dense

    open(file, "a") do io
        println(io, N, ",", n, ",", p, ",", c, ",", d, ",", false, ",", t1)
        println(io, N, ",", n, ",", p, ",", c, ",", d, ",", true, ",", t2)
    end
end

# overdetermined
nsamples = 500
n = nsamples*c
for d in d_vals # go hard -> easy
    data = MVDA.simulate_WS2007(n, p, c, nsamples, d)
    standardize!(data[2])
    idx = randperm(MersenneTwister(16), n)
    _permute!(data[1], idx)
    _permutecols!(data[2], idx)

    (cv_set, test_set) = splitobs(data, at=0.8, obsdim=1)
    filename = "overdetermined-d=$(d)"
    title = "d = $(d)"
    t1 = run("$(filename)-path=D2S", title, cv_set, test_set, false) # dense -> sparse
    t2 = run("$(filename)-path=S2D", title, cv_set, test_set, true)  # sparse -> dense

    open(file, "a") do io
        println(io, N, ",", n, ",", p, ",", c, ",", d, ",", false, ",", t1)
        println(io, N, ",", n, ",", p, ",", c, ",", d, ",", true, ",", t2)
    end
end
