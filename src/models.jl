#
#   This defines types and dispatch rules for various terms in a VDA model.
#
#   Types: All concrete types are singleton types; i.e. no fields.
#
#   - AbstractVDALoss: A loss function. Evaluation should ALWAYS handle a VDA
#       loss first. 
#
#   - AbstractVDAPenalty: A penalty function. Evaluation should ALWAYS accumulate
#       contributions from a penalty.
#
#   - SquaredDistancePenalty: Distance penalty used by proximal distance algorithms.
#       Rules rely on dispatch on projection operator to determine scale factors.
#       The projection operator must come from the extras datastructure.
#
#   - AbstractVDAModel: An objective function composed of a loss and possibly one
#       or more penalties.
#
#   Functions:
#
#   - 4-arg get_scale_factor(): Use dispatch to select correct evaluation rule and
#       forward any datastructures needed to compute factor.
#
#   - 3-arg get_scale_factor(): Implementation of calculation rule. The stength of
#       penalty terms (e.g. lambda) SHOULD NOT be included in the scale factor.
# 
#   - 4-arg evaluate_model!(): Use dispatch to collect scale factor(s) and populate
#       work buffers. This includes evaluating residuals and projections.
#
#   - 3-arg evaluate_model!(): Implements evaluation of terms. Second argument is a
#       Tuple containing constants and work buffers required for evaluation.    
#

abstract type AbstractVDAPenalty end

function get_scale_factor(f::AbstractVDAPenalty, problem::MVDAProblem, extras, hparams)
    get_scale_factor(f, problem.coeff.slope, hparams)
end

function evaluate_model!(f::AbstractVDAPenalty, problem::MVDAProblem, extras, hparams)
    B, G = problem.coeff, problem.grad
    scale_factor = get_scale_factor(f, problem, extras, hparams)
    evaluate_model!(f, (scale_factor, B, G), hparams)
end

struct RidgePenalty <: AbstractVDAPenalty end

function get_scale_factor(::RidgePenalty, B, hparams)
    scale_factor = 1 / size(B, 1)
    return scale_factor
end

function evaluate_model!(::RidgePenalty, (scale_factor, B, G), hparams)
    # gradient
    scaled_lambda = hparams.lambda*scale_factor
    BLAS.axpy!(scaled_lambda, B.slope, G.slope)
    return scale_factor * dot(B.slope, B.slope)
end

struct LassoPenalty <: AbstractVDAPenalty end

function get_scale_factor(::LassoPenalty, B, hparams)
    scale_factor = 1 / prod(size(B))
    return scale_factor
end

function evaluate_model!(::LassoPenalty, (scale_factor, B, G), hparams)
    # gradient
    scaled_lambda = hparams.lambda * scale_factor
    for jl in eachindex(B.slope)
        bjl = B.slope[jl]
        v = ifelse(bjl < 0, -scaled_lambda, scaled_lambda)
        G.slope[jl] += v
    end
    return scale_factor * norm(B.slope, 1)
end

# struct ElasticNetPenalty <: AbstractVDAPenalty end

# function evaluate_model!(::ElasticNetPenalty, problem::MVDAProblem, extras, hparams)
#     @unpack alpha = hparams
#     penalty1 = evaluate_model!(LassoPenalty(), problem, extras, hparams)
#     penalty2 = evaluate_model!(RidgePenalty(), problem, extras, hparams)

#     return (1-alpha)*penalty1 + alpha*penalty2
# end

struct SquaredDistancePenalty <: AbstractVDAPenalty end

function get_scale_factor(::SquaredDistancePenalty, problem::MVDAProblem, extras, hparams)
    get_scale_factor(extras.projection, problem.coeff.slope, hparams)
end

function evaluate_model!(::SquaredDistancePenalty, problem::MVDAProblem, extras, hparams)
    G, R = problem.grad, problem.res
    scale_factor = get_scale_factor(SquaredDistancePenalty(), problem, extras, hparams)
    apply_projection(problem, extras, hparams, false)
    evaluate_residuals!(problem, extras, hparams.epsilon, false, true)
    evaluate_model!(SquaredDistancePenalty(), (scale_factor, G, R), hparams)
end

function evaluate_model!(::SquaredDistancePenalty, (scale_factor, G, R), hparams)
    # gradient
    scaled_rho = hparams.rho * scale_factor
    BLAS.axpy!(-scaled_rho, R.dist, G.slope)
    return scale_factor * dot(R.dist, R.dist)
end

abstract type AbstractVDALoss end

struct SquaredEpsilonInsensitiveLoss <: AbstractVDALoss end

function evaluate_model!(::SquaredEpsilonInsensitiveLoss, problem::MVDAProblem, extras, hparams)
    @unpack epsilon = hparams
    A, G, R, intercept = design_matrix(problem), problem.grad, problem.res, problem.intercept
    T = floattype(problem)
    n, _, _ = probsizes(problem)
    alpha, beta = convert(T, 1/n), zero(T)
    evaluate_residuals!(problem, extras, epsilon, true, false)
    evaluate_model!(SquaredEpsilonInsensitiveLoss(), (alpha, beta, A, G, R, intercept), hparams)
end

function evaluate_model!(::SquaredEpsilonInsensitiveLoss, (alpha, beta, A, G, R, intercept), hparams)
    # gradient
    fillzero = Base.Fix2(fill!, zero(beta))
    foreach(fillzero, G)
    if intercept
        mean!(G.intercept, transpose(R.loss))
        G.intercept .*= -one(beta)
    end
    BLAS.gemm!('T', 'N', -alpha, A, R.loss, beta, G.slope)
    return alpha * dot(R.loss, R.loss)
end

const SqDistPenalty = SquaredDistancePenalty
const SqEpsilonLoss = SquaredEpsilonInsensitiveLoss
# const EpsilonLoss = EpsilonInsensitiveLoss



abstract type AbstractVDAModel end

struct UnpenalizedObjective{LOSS} <: AbstractVDAModel end

function UnpenalizedObjective(::T) where {T<:AbstractVDALoss}
    UnpenalizedObjective{T}()
end

function get_scale_factor(::UnpenalizedObjective{T}, problem::MVDAProblem, extras, hparams) where T
    1.0
end

function evaluate_model!(::UnpenalizedObjective{LOSS}, problem::MVDAProblem, extras, hparams) where LOSS <: AbstractVDALoss
    #
    G = problem.grad

    risk = evaluate_model!(LOSS(), problem, extras, hparams)
    penalty = 0.0
    loss = 1//2*risk
    distsq = 0.0
    obj = loss
    gradsq = dot(G, G)

    return __eval_result__(risk, loss, obj, penalty, distsq, gradsq)
end

struct PenalizedObjective{LOSS,PENALTY} <: AbstractVDAModel end

function PenalizedObjective(::T1, ::T2) where {T1<:AbstractVDALoss,T2<:AbstractVDAPenalty}
    PenalizedObjective{T1,T2}()
end

function get_scale_factor(::PenalizedObjective{T1,T2}, problem::MVDAProblem, extras, hparams) where {T1,T2}
    get_scale_factor(T2(), problem, extras, hparams)
end

function evaluate_model!(::PenalizedObjective{SqEpsilonLoss,SqDistPenalty}, problem::MVDAProblem, extras, hparams)
    #
    @unpack rho = hparams
    G = problem.grad

    risk = evaluate_model!(SqEpsilonLoss(), problem, extras, hparams)
    penalty = evaluate_model!(SqDistPenalty(), problem, extras, hparams)
    loss = 1//2*risk
    distsq = penalty
    obj = loss + 1//2*rho*distsq
    gradsq = dot(G, G)

    return __eval_result__(risk, loss, obj, penalty, distsq, gradsq)
end

function evaluate_model!(::PenalizedObjective{SqEpsilonLoss,RidgePenalty}, problem::MVDAProblem, extras, hparams)
    #
    @unpack lambda = hparams
    G = problem.grad

    risk = evaluate_model!(SqEpsilonLoss(), problem, extras, hparams)
    penalty = evaluate_model!(RidgePenalty(), problem, extras, hparams)
    loss = 1//2*risk + 1//2*lambda*penalty
    distsq = zero(risk)
    obj = loss
    gradsq = dot(G, G)

    return __eval_result__(risk, loss, obj, penalty, distsq, gradsq)
end

function evaluate_model!(::PenalizedObjective{SqEpsilonLoss,LassoPenalty}, problem::MVDAProblem, extras, hparams)
    #
    @unpack lambda = hparams
    G = problem.grad

    risk = evaluate_model!(SqEpsilonLoss(), problem, extras, hparams)
    penalty = evaluate_model!(LassoPenalty(), problem, extras, hparams)
    loss = 1//2*risk + lambda*penalty
    distsq = zero(risk)
    obj = loss
    gradsq = dot(G, G)

    return __eval_result__(risk, loss, obj, penalty, distsq, gradsq)
end
