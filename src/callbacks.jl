"""
Placeholder for callbacks in main functions.
"""
__do_nothing_callback__((iter, state), problem, hyperparams) = nothing
# __do_nothing_callback__(statistics, problem, hyperparams, indices) = nothing
# __do_nothing_callback__(::Int) = __do_nothing_callback__

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
