using Revise
using MVDA, Plots, Statistics, Random, LinearAlgebra

include("cv-driver.jl")

N = 100
c = 4
p = 100
ϵ = 0.5 * sqrt(2*c/(c-1))
d_vals = (1.0, 2.0, 3.0)

file = "/home/alanderos/Desktop/VDA/WS2007-timings.txt"
isfile(file) && rm(file)
open(file, "a") do io
    println(io, "replicates,n,p,c,d,sparse2dense,time")
end

# underdetermined
nsamples = 20
n = nsamples*c
for d in d_vals # go hard -> easy
    targets, X = MVDA.simulate_WS2007(n, p, c, nsamples, d)
    filename = "underdetermined-d=$(d)"
    title = "d = $(d)"
    t1 = run_example(MersenneTwister(123456), "$(filename)-path=D2S", title, targets, X, N, false) # dense -> sparse
    t2 = run_example(MersenneTwister(123456), "$(filename)-path=S2D", title, targets, X, N, true)  # sparse -> dense

    open(file, "a") do io
        println(io, N, ",", n, ",", p, ",", c, ",", d, ",", false, ",", t1)
        println(io, N, ",", n, ",", p, ",", c, ",", d, ",", true, ",", t2)
    end
end

# overdetermined
nsamples = 500
n = nsamples*c
for d in d_vals # go hard -> easy
    targets, X = MVDA.simulate_WS2007(n, p, c, nsamples, d)
    filename = "overdetermined-d=$(d)"
    title = "d = $(d)"
    t1 = run_example(MersenneTwister(123456), "$(filename)-path=D2S", title, targets, X, N, false) # dense -> sparse
    t2 = run_example(MersenneTwister(123456), "$(filename)-path=S2D", title, targets, X, N, true)  # sparse -> dense

    open(file, "a") do io
        println(io, N, ",", n, ",", p, ",", c, ",", d, ",", false, ",", t1)
        println(io, N, ",", n, ",", p, ",", c, ",", d, ",", true, ",", t2)
    end
end
