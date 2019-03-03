include("precond.jl")
include("stepsize.jl")

######################
### Mutable states ###
######################

mutable struct ThreePhaseState{T<:Integer}
    n           :: T
    window_size :: T
    next_window :: T
end

################
### Adapterers ###
################

abstract type CompositeAdapter <: AbstractAdapter end

struct NaiveCompAdapter{Tpc <: PreConditioner, Tssa <: StepSizeAdapter} <: CompositeAdapter
    pc  :: Tpc
    ssa :: Tssa
end

function getss(tp::CompositeAdapter)
    return getss(tp.ssa)
end

# Acknowledgement: this adaption settings is mimicing Stan's 3-phase adaptation.
struct ThreePhaseAdapter{T<:Integer, Tpc <: PreConditioner, Tssa <: StepSizeAdapter} <: CompositeAdapter
    n_adapts    :: T
    init_buffer :: T
    term_buffer :: T
    pc          :: Tpc
    ssa         :: Tssa
    state       :: ThreePhaseState{T}
end

function get_threephase_params(::Nothing)
    init_buffer, term_buffer, window_size = 75, 50, 25
    next_window = init_buffer + window_size - 1
    return init_buffer, term_buffer, window_size, next_window
end

function ThreePhaseAdapter(spl::Sampler{<:AdaptiveHamiltonian}, ϵ::Real, dim::Integer)
    # Diagonal pre-conditioner
    # pc = UnitPreConditioner()
    pc = DiagPreConditioner(dim)
    # pc = DensePreConditioner(dim)
    # Dual averaging for step size
    ssa = DualAveraging(spl, spl.info[:adapt_conf], ϵ)
    # Window parameters
    init_buffer, term_buffer, window_size, next_window = get_threephase_params(spl.info[:adapt_conf])
    threephasestate = ThreePhaseState(0, window_size, next_window)
    return ThreePhaseAdapter(spl.alg.n_adapts, init_buffer, term_buffer, pc, ssa, threephasestate)
end

# Ref: https://github.com/stan-dev/stan/blob/develop/src/stan/mcmc/windowed_adaptation.hpp
function in_adaptation(tp::ThreePhaseAdapter)
    return (tp.state.n >= tp.init_buffer) &&
           (tp.state.n < tp.n_adapts - tp.term_buffer) &&
           (tp.state.n != tp.n_adapts)
end

function is_windowend(tp::ThreePhaseAdapter)
    return (tp.state.n == tp.state.next_window) &&
           (tp.state.n != tp.n_adapts)
end

function compute_next_window!(tp::ThreePhaseAdapter)
    if ~(tp.state.next_window == tp.n_adapts - tp.term_buffer - 1)
        tp.state.window_size *= 2
        tp.state.next_window = tp.state.n + tp.state.window_size
        if ~(tp.state.next_window == tp.n_adapts - tp.term_buffer - 1)
            next_window_boundary = tp.state.next_window + 2 * tp.state.window_size
            if (next_window_boundary >= tp.n_adapts - tp.term_buffer)
                tp.state.next_window = tp.n_adapts - tp.term_buffer - 1
            end
        end
    end
end

function adapt!(tp::ThreePhaseAdapter, stats::Real, θ; adapt_ϵ=false, adapt_M=false)
    if tp.state.n < tp.n_adapts
        tp.state.n += 1
        if tp.state.n == tp.n_adapts
            if adapt_ϵ
                ϵ = exp(tp.ssa.state.x_bar)
                tp.ssa.state.ϵ = min(one(ϵ), ϵ)
            end
            @info " Adapted ϵ = $(getss(tp)), std = $(string(tp.pc)); $(tp.state.n) iterations is used for adaption."
        else
            if adapt_ϵ
                is_updateμ = is_windowend(tp)# || tp.state.n == tp.n_adapts
                adapt!(tp.ssa, stats, is_updateμ)
            end

            # Ref: https://github.com/stan-dev/stan/blob/develop/src/stan/mcmc/hmc/nuts/adapt_diag_e_nuts.hpp
            if adapt_M
                is_addsample, is_updatestd = in_adaptation(tp), is_windowend(tp)
                adapt!(tp.pc, θ, is_addsample, is_updatestd)
            end

            # If window ends, compute next window
            is_windowend(tp) && compute_next_window!(tp)
        end
    end
end
