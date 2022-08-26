"""
Placeholder for callbacks in main functions.
"""
__do_nothing_callback__((iter, state), problem, hyperparams) = nothing
__do_nothing_callback__(statistics, problem, hyperparams, indices) = nothing

function prediction_accuracies(problem::MVDAProblem, train_set, validation_set, test_set)
    Tr = accuracy(problem, train_set)
    V = accuracy(problem, validation_set)
    T = accuracy(problem, test_set)

    return (Tr, V, T)
end

function prediction_errors(problem, train_set, validation_set, test_set)
    Tr, V, T = prediction_accuracies(problem, train_set, validation_set, test_set)
    return (1-Tr, 1-V, 1-T)
end

struct VerboseCallback <: Function
    every::Int
end

VerboseCallback() = VerboseCallback(1)

function (F::VerboseCallback)((iter, state), problem::MVDAProblem, hyperparams)
    if iter == 0
        @printf("\n%-5s\t%-8s\t%-8s\t%-8s\t%-8s\t%-12s\t%-8s\t%-8s\n", "iter", "rho", "risk", "loss", "objective", "penalty", "|gradient|", "distance")
    end
    if iter % F.every == 0
        @printf("%4d\t%4.3e\t%4.3e\t%4.3e\t%4.3e\t%8.3e\t%4.3e\t%4.3e\n", iter, hyperparams.rho, state.risk, state.loss, state.objective, state.penalty, state.gradient, state.distance)
    end

    return nothing
end

struct CVCallback{T,N} <: Function
    dims::NTuple{N,Int}
    data::Dict{Symbol,Array{T,N}}

    function CVCallback{T}(dims::NTuple{N,Int}) where {T,N}
        data = Dict{Symbol,Array{T,N}}()
        new{T,N}(dims, data)
    end
end

CVCallback{T}(dims...) where T = CVCallback{T}(dims)    # call inner constructor
CVCallback(dims...) = CVCallback{Float64}(dims...)      # default to Float64 eltype

const VALID_FIELDS = [
    :gamma, :epsilon, :lambda, :sparsity,   # hyperparameters; probably not needed
    :iters,                                 # total number of iterations
    :risk, :loss, :objective,               # loss metrics
    :gradient, :distance, :penalty,         # convergence quality metrics
    :nactive, :pactive,                     # number of active features
]

function add_field!(cb::CVCallback, field::Symbol)
    global VALID_FIELDS
    if !(field in VALID_FIELDS)
        error("Unknown metric $(field).")
    end
    dims, data = cb.dims, cb.data
    T = valtype(data)
    data[field] = T(undef, dims)
    return data
end

function add_field!(cb::CVCallback, fields::Vararg{Symbol,N}) where N
    for field in fields
        add_field!(cb, field)
    end
    return cb.data
end

function (F::CVCallback{<:Any,4})(statistics::Tuple, problem::MVDAProblem, hyperparams, indices)
    @unpack data = F
    i, j, l, k = values(indices)
    for (field, arr) in data
        arr[i,j,l,k] = _get_statistic_(statistics, problem, hyperparams, Val(field))
    end
end

function (F::CVCallback{<:Any,3})(statistics::Tuple, problem::MVDAProblem, hyperparams, indices)
    @unpack data = F
    i, j, k = values(indices)
    for (field, arr) in data
        arr[i,j,k] = _get_statistic_(statistics, problem, hyperparams, Val(field))
    end
end

function (F::CVCallback{<:Any,2})(statistics::Tuple, problem::MVDAProblem, hyperparams, indices)
    @unpack data = F
    i, k = values(indices)
    for (field, arr) in data
        arr[i,k] = _get_statistic_(statistics, problem, hyperparams, Val(field))
    end
end

_get_statistic_(::Any, ::Any, hyperparams, ::Val{:gamma}) = hyperparams.gamma
_get_statistic_(::Any, ::Any, hyperparams, ::Val{:epsilon}) = hyperparams.epsilon
_get_statistic_(::Any, ::Any, hyperparams, ::Val{:lambda}) = hyperparams.lambda
_get_statistic_(::Any, ::Any, hyperparams, ::Val{:sparsity}) = hyperparams.sparsity

_get_statistic_(statistics, ::Any, ::Any, ::Val{:iters}) = first(statistics)
_get_statistic_(statistics, ::Any, ::Any, ::Val{:risk}) = last(statistics).risk
_get_statistic_(statistics, ::Any, ::Any, ::Val{:loss}) = last(statistics).loss
_get_statistic_(statistics, ::Any, ::Any, ::Val{:objective}) = last(statistics).objective
_get_statistic_(statistics, ::Any, ::Any, ::Val{:distance}) = last(statistics).distance
_get_statistic_(statistics, ::Any, ::Any, ::Val{:penalty}) = last(statistics).penalty
_get_statistic_(statistics, ::Any, ::Any, ::Val{:gradient}) = last(statistics).gradient

_get_statistic_(::Any, problem::MVDAProblem, ::Any, ::Val{:nactive}) = count_active_variables(problem)
_get_statistic_(::Any, problem::MVDAProblem, ::Any, ::Val{:pactive}) = count_active_variables(problem) / size(problem.coeff.slope, 1)


struct HistoryCallback{T} <: Function
    data::Dict{Symbol,Vector{T}}

    function HistoryCallback{T}() where T
        data = Dict{Symbol,Vector{T}}()
        new{T}(data)
    end
end

HistoryCallback() = HistoryCallback{Float64}()    # default to Float64 eltype

function add_field!(cb::HistoryCallback, field::Symbol)
    global VALID_FIELDS
    if !(field in VALID_FIELDS)
        error("Unknown metric $(field).")
    end
    data = cb.data
    T = valtype(data)
    data[field] = T[]
    return data
end

function add_field!(cb::HistoryCallback, fields::Vararg{Symbol,N}) where N
    for field in fields
        add_field!(cb, field)
    end
    return cb.data
end

function (F::HistoryCallback)(statistics::Tuple, problem::MVDAProblem, hyperparams)
    @unpack data = F
    for (field, arr) in data
        push!(arr, _get_statistic_(statistics, problem, hyperparams, Val(field)))
    end
end
