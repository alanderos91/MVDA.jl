using Revise
using MVDA, Plots, Statistics, Random, LinearAlgebra

include("cv-driver.jl")

function simulate(n, p, c, nsamples, overlap)
    # simulate c classes with single predictor as in Wang and Shen 2007
    X = randn(n, p)
    for i in 1:c
        idx = nsamples*(i-1)+1:nsamples*i
        a₁ = overlap * cos(2*(π/6 + (i-1)*π/c))
        a₂ = overlap * sin(2*(π/6 + (i-1)*π/c))
        X[idx,1] .+= a₁
        X[idx,2] .+= a₂
    end
    targets = zeros(Int, n)
    for j in 1:c
        targets[nsamples*(j-1)+1:nsamples*j] .= j
    end
    idx = randperm(n)
    targets, X = targets[idx], X[idx,:]
    return targets, X
end

N = 100
c = 4
p = 100
ϵ = 0.5 * sqrt(2*c/(c-1))
d_vals = (1.0, 2.0, 3.0)

file = "/home/alanderos/Desktop/timings.txt"
isfile(file) && rm(file)
open(file, "a") do io
    println(io, "replicates,n,p,c,d,sparse2dense,time")
end

# underdetermined
nsamples = 20
n = nsamples*c
for d in d_vals # go hard -> easy
    targets, X = simulate(n, p, c, nsamples, d)
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
    targets, X = simulate(n, p, c, nsamples, d)
    filename = "overdetermined-d=$(d)"
    title = "d = $(d)"
    t1 = run_example(MersenneTwister(123456), "$(filename)-path=D2S", title, targets, X, N, false) # dense -> sparse
    t2 = run_example(MersenneTwister(123456), "$(filename)-path=S2D", title, targets, X, N, true)  # sparse -> dense

    open(file, "a") do io
        println(io, N, ",", n, ",", p, ",", c, ",", d, ",", false, ",", t1)
        println(io, N, ",", n, ",", p, ",", c, ",", d, ",", true, ",", t2)
    end
end
