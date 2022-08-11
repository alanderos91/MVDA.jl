module MVDA

# Dataset Management and I/O
using DataFrames: copy, copyto!
using CodecZlib, CSV, DataDeps, DataFrames, MLDataUtils, Printf, ProgressMeter

# Math & Stats
using Parameters
using KernelFunctions, LinearAlgebra, Random, StableRNGs, Statistics, StatsBase

# Multithreading
using Polyester

# Imports
import Base: show, iterate
import MLDataUtils: classify, predict

##### DATA #####

#=
Uses DataDeps to download data as needed.
Inspired by UCIData.jl: https://github.com/JackDunnNZ/UCIData.jl
=#

##### DATADEP REGISTRATION #####

const DATADEPNAME = "MVDA"
const DATA_DIR = joinpath(@__DIR__, "data")
const MESSAGES = Ref(String[])
const REMOTE_PATHS = Ref([])
const CHECKSUMS = Ref([])
const FETCH_METHODS = Ref([])
const POST_FETCH_METHODS = Ref([])
const DATASETS = Ref(String[])

include("simulation.jl")
include("datadeps.jl")

function __init__()
    # Delete README.md in data/.
    readme = joinpath(DATA_DIR, "README.md")
    if ispath(readme)
        rm(readme)
    end
    
    # Add arguments to Refs.
    for dataset_jl in readdir(DATA_DIR)
        include(joinpath(DATA_DIR, dataset_jl))
    end

    # Compile a help message from each dataset.
    # Save the output of the help message in DATA_DIR.
    readme_content = """
    # MVDA Examples
    
    You can load an example by invoking `MVDA.dataset(name)`.
    The list of available datasets is accessible via `MVDA.list_datasets()`.

    Please note that the descriptions here are *very* brief summaries. Follow the links for additional information.

    $(join(MESSAGES[], '\n')) 
    """

    open(readme, "w") do io
        write(io, readme_content)
    end

    # Register the DataDep as MVDA.
    register(DataDep(
        DATADEPNAME,
        """
        Welcome to the MVDA installation.
    
        This program will now attempt to

            (1) download a few datasets from the UCI Machine Learning Repository, and
            (2) simulate additional synthetic datasets.
        
        Please see $(readme) for a preview of each example.
        """,
        REMOTE_PATHS[],
        CHECKSUMS[];
        fetch_method=FETCH_METHODS[],
        post_fetch_method=POST_FETCH_METHODS[],
    ))

    # Trigger the download process.
    @datadep_str(DATADEPNAME)
end

##### END DATADEP REGISTRATION #####

##### IMPLEMENTATION #####

include("problem.jl")
include("utilities.jl")
include("projections.jl")

abstract type AbstractMMAlg end

include(joinpath("algorithms", "SD.jl"))
include(joinpath("algorithms", "MMSVD.jl"))
include(joinpath("algorithms", "CyclicVDA.jl"))

const DEFAULT_ANNEALING = geometric_progression
const DEFAULT_CALLBACK = __do_nothing_callback__
const DEFAULT_SCORE_FUNCTION = prediction_error

include("fit.jl")
include("cv.jl")

export IterationResult, SubproblemResult
export MVDAProblem, SD, MMSVD # CyclicVDA

end
