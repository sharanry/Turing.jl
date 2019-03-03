"""
    HMC(n_iters::Int, epsilon::Float64, tau::Int)

Hamiltonian Monte Carlo sampler.

Arguments:

- `n_iters::Int` : The number of samples to pull.
- `epsilon::Float64` : The leapfrog step size to use.
- `tau::Int` : The number of leapfrop steps to use.

Usage:

```julia
HMC(1000, 0.05, 10)
```

Example:

```julia
# Define a simple Normal model with unknown mean and variance.
@model gdemo(x) = begin
    s ~ InverseGamma(2,3)
    m ~ Normal(0, sqrt(s))
    x[1] ~ Normal(m, sqrt(s))
    x[2] ~ Normal(m, sqrt(s))
    return s, m
end

sample(gdemo([1.5, 2]), HMC(1000, 0.05, 10))
```

Tips:

- If you are receiving gradient errors when using `HMC`, try reducing the
`step_size` parameter.

```julia
# Original step_size
sample(gdemo([1.5, 2]), HMC(1000, 0.1, 10))

# Reduced step_size.
sample(gdemo([1.5, 2]), HMC(1000, 0.01, 10))
```
"""
mutable struct HMC{AD, space} <: StaticHamiltonian{AD}
    n_iters   ::  Int       # number of samples
    epsilon   ::  Float64   # leapfrog step size
    tau       ::  Int       # leapfrog step number
    gid       ::  Int       # group ID
end
HMC(args...) = HMC{ADBackend()}(args...)
function HMC{AD}(epsilon::Float64, tau::Int, space...) where AD
    return HMC{AD, space}(1, epsilon, tau, 0)
end
function HMC{AD}(n_iters::Int, epsilon::Float64, tau::Int) where AD
    return HMC{AD, ()}(n_iters, epsilon, tau, 0)
end
function HMC{AD}(n_iters::Int, epsilon::Float64, tau::Int, space...) where AD
    return HMC{AD, space}(n_iters, epsilon, tau, 0)
end
function HMC{AD1}(alg::HMC{AD2, space}, new_gid::Int) where {AD1, AD2, space}
    return HMC{AD1, space}(alg.n_iters, alg.epsilon, alg.tau, new_gid)
end
function HMC{AD, space}(alg::HMC, new_gid::Int) where {AD, space}
    return HMC{AD, space}(alg.n_iters, alg.epsilon, alg.tau, new_gid)
end

# Below is a trick to remove the dependency of Stan by Requires.jl
# Please see https://github.com/TuringLang/Turing.jl/pull/459 for explanations
DEFAULT_ADAPT_CONF_TYPE = Nothing
STAN_DEFAULT_ADAPT_CONF = nothing

#Sampler(model, alg::Union{HMC, HMCDA}, vi::AbstractVarInfo) = Sampler(model, alg, vi, nothing, 0)
function Sampler(model, alg::Union{HMC, HMCDA}, vi::AbstractVarInfo, adapt_conf, eval_num)
    spl = Sampler(alg, nothing)
    idcs = VarReplay._getidcs(vi, spl)
    ranges = VarReplay._getranges(vi, spl, idcs)
    info = HMCInfo(model, spl, vi, adapt_conf, idcs, ranges, eval_num)
    return Sampler(alg, info)
end

mutable struct HMCInfo{Tidcs, Tranges, Tconf, Twum}
    idcs::Tidcs
    cache_updated::UInt8
    ranges::Tranges
    eval_num::Int
    adapt_conf::Tconf
    progress::ProgressMeter.Progress
    wum::Twum
    lf_num::Int
end
function HMCInfo(model, spl, vi, adapt_conf, idcs, ranges, eval_num)
    alg = spl.alg
    wum = init_adapter(model, spl, vi, adapt_conf)
    return HMCInfo( idcs, 
                    CACHERESET, 
                    ranges, 
                    eval_num, 
                    adapt_conf, 
                    ProgressMeter.Progress(alg.n_iters, 1, "[HMC] Sampling...", 0), 
                    wum,
                    0
                    )
end

function hmc_step(θ, lj, lj_func, grad_func, H_func, ϵ, alg::HMC, momentum_sampler::Function;
                  rev_func=nothing, log_func=nothing)
    θ_new, lj_new, is_accept, τ_valid, α = _hmc_step(
                θ, lj, lj_func, grad_func, H_func, alg.tau, ϵ, momentum_sampler; rev_func=rev_func, log_func=log_func)
    return θ_new, lj_new, is_accept, α
end

function init_spl(model, alg::Union{HMC, HMCDA}; 
                            reuse_spl_n = 0, 
                            resume_from = nothing, 
                            adapt_conf=STAN_DEFAULT_ADAPT_CONF, 
                            kwargs...,
                    )

    if resume_from == nothing
        eval_num = 1
        vi = TypedVarInfo(default_varinfo(model))
    else
        vi = deepcopy(resume_from.info.vi)
    end
    
    if reuse_spl_n > 0
        spl = resume_from.info.spl
    else
        spl = Sampler(model, alg, vi, adapt_conf, eval_num)
    end
    @assert isa(spl.alg, Hamiltonian) "[Turing] alg type mismatch; please use resume() to re-use spl"

    return spl, vi
end

function get_sample_n(alg::Hamiltonian; reuse_spl_n = 0, kwargs...)
    return reuse_spl_n > 0 ? reuse_spl_n : alg.n_iters
end

function init_varinfo(model, spl::Sampler{<:Hamiltonian}; resume_from = nothing, kwargs...)
    if resume_from == nothing
        spl.info.eval_num += 1
        return TypedVarInfo(default_varinfo(model))
    else
        return deepcopy(resume_from.info.vi)
    end
end

macro loop_iter(i)
    return esc(quote
        Turing.DEBUG && @debug "$alg_str stepping..."

        time_elapsed = @elapsed vi, is_accept = step(model, spl, vi, Val($(i == 1)))
        time_total += time_elapsed

        if is_accept # accepted => store the new predcits
            samples[$i].value = Sample(vi, spl).value
        else         # rejected => store the previous predcits
            samples[$i] = samples[$i - 1]
        end
        samples[$i].info.elapsed = time_elapsed
        if isdefined(spl.info, :wum)
            samples[$i].info.lf_eps = getss(spl.info.wum)
        end

        total_lf_num += spl.info.lf_num
        total_eval_num += spl.info.eval_num
        push!(accept_his, is_accept)
        PROGRESS[] && ProgressMeter.next!(spl.info.progress)
    end)
end

function _sample(vi, samples, spl, model, alg::Hamiltonian,
                                chunk_size=CHUNKSIZE[],             # set temporary chunk size
                                save_state=false,                   # flag for state saving
                                resume_from=nothing,                # chain to continue
                                reuse_spl_n=0,                      # flag for spl re-using
                                adapt_conf=STAN_DEFAULT_ADAPT_CONF, # adapt configuration
                )
    if ADBACKEND[] == :forward_diff
        default_chunk_size = CHUNKSIZE[]  # record global chunk size
        setchunksize(chunk_size)        # set temp chunk size
    end

    alg_str = isa(alg, HMC)   ? "HMC"   :
              isa(alg, HMCDA) ? "HMCDA" :
              isa(alg, SGHMC) ? "SGHMC" :
              isa(alg, SGLD)  ? "SGLD"  :
              isa(alg, NUTS)  ? "NUTS"  : "Hamiltonian"

    # Initialization
    time_total = zero(Float64)

    if spl.alg.gid == 0
        link!(vi, spl)
        runmodel!(model, vi, spl)
    end

    # HMC steps
    total_lf_num = 0
    total_eval_num = 0
    accept_his = Bool[]
    n = length(samples)
    PROGRESS[] && (spl.info.progress = ProgressMeter.Progress(n, 1, "[$alg_str] Sampling...", 0))
    @loop_iter(1)
    for i = 2:n
        @loop_iter(i)
    end

    println("[$alg_str] Finished with")
    println("  Running time        = $time_total;")
    if ~isa(alg, NUTS)  # accept rate for NUTS is meaningless - so no printing
        accept_rate = sum(accept_his) / n  # calculate the accept rate
        println("  Accept rate         = $accept_rate;")
    end
    println("  #lf / sample        = $(total_lf_num / n);")
    println("  #evals / sample     = $(total_eval_num / n);")
    if isdefined(spl.info, :wum)
      std_str = string(spl.info.wum.pc)
      std_str = length(std_str) >= 32 ? std_str[1:30]*"..." : std_str   # only show part of pre-cond
      println("  pre-cond. metric    = $(std_str).")
    end

    if ADBACKEND[] == :forward_diff
        setchunksize(default_chunk_size)      # revert global chunk size
    end

    if resume_from != nothing   # concat samples
        pushfirst!(samples, resume_from.value2...)
    end
    c = Chain(0.0, samples)       # wrap the result by Chain
    if save_state               # save state
        # Convert vi back to X if vi is required to be saved
        if spl.alg.gid == 0 invlink!(vi, spl) end
        save!(c, spl, model, vi)
    end
    return c
end

function step(model, spl::Sampler{<:StaticHamiltonian}, vi::AbstractVarInfo, is_first::Val{true})
    spl.info.wum = NaiveCompAdapter(UnitPreConditioner(), FixedStepSize(spl.alg.epsilon))
    return vi, true
end

function init_adapter(model, spl::Sampler{<:StaticHamiltonian}, vi, adapt_conf=nothing)
    return NaiveCompAdapter(UnitPreConditioner(), FixedStepSize(spl.alg.epsilon))
end

function init_adapter(model, spl::Sampler{<:AdaptiveHamiltonian}, vi, adapt_conf)
    epsilon = find_good_eps(model, spl, vi) # heuristically find good initial epsilon
    dim = length(vi[spl])
    return ThreePhaseAdapter(spl, epsilon, dim, adapt_conf)
end

function step(model, spl::Sampler{<:AdaptiveHamiltonian}, vi::AbstractVarInfo, is_first::Val{true})
    spl.alg.gid != 0 && link!(vi, spl)
    spl.info.wum = init_adapter(model, spl, vi, spl.info.adapt_conf)
    spl.alg.gid != 0 && invlink!(vi, spl)
    return vi, true
end

function step(model, spl::Sampler{<:Hamiltonian}, vi::AbstractVarInfo, is_first::Val{false})
    # Get step size
    ϵ = getss(spl.info.wum)
    Turing.DEBUG && @debug "current ϵ: $ϵ"

    # Reset current counters
    spl.info.lf_num = 0
    spl.info.eval_num = 0

    Turing.DEBUG && @debug "X-> R..."
    if spl.alg.gid != 0
        link!(vi, spl)
        runmodel!(model, vi, spl)
    end

    grad_func = gen_grad_func(vi, spl, model)
    lj_func = gen_lj_func(vi, spl, model)
    rev_func = gen_rev_func(vi, spl)
    log_func = gen_log_func(spl)
    momentum_sampler = gen_momentum_sampler(vi, spl, spl.info.wum.pc)
    H_func = gen_H_func(spl.info.wum.pc)

    θ, lj = vi[spl], vi.logp

    θ_new, lj_new, is_accept, α = hmc_step(θ, lj, lj_func, grad_func, H_func, ϵ, spl.alg, momentum_sampler;
                                           rev_func=rev_func, log_func=log_func)

    Turing.DEBUG && @debug "decide whether to accept..."
    if is_accept
        vi[spl] = θ_new
        setlogp!(vi, lj_new)
    else
        vi[spl] = θ
        setlogp!(vi, lj)
    end

    if PROGRESS[] && spl.alg.gid == 0
        std_str = string(spl.info.wum.pc)
        std_str = length(std_str) >= 32 ? std_str[1:30]*"..." : std_str
        isdefined(spl.info, :progress) && ProgressMeter.update!(
            spl.info.progress,
            spl.info.progress.counter;
            showvalues = [(:ϵ, ϵ), (:α, α), (:pre_cond, std_str)],
        )
    end

    if spl.alg isa AdaptiveHamiltonian
        adapt!(spl.info.wum, α, vi[spl], adapt_M=false, adapt_ϵ=true)
    end

    Turing.DEBUG && @debug "R -> X..."
    spl.alg.gid != 0 && invlink!(vi, spl)

    return vi, is_accept
end

function assume(spl::Sampler{<:Hamiltonian}, dist::Distribution, vn::VarName, vi::AbstractVarInfo)
    Turing.DEBUG && @debug "assuming..."
    updategid!(vi, vn, spl)
    r = vi[vn]
    # acclogp!(vi, logpdf_with_trans(dist, r, istrans(vi, vn)))
    # r
    Turing.DEBUG && @debug "dist = $dist"
    Turing.DEBUG && @debug "vn = $vn"
    Turing.DEBUG && @debug "r = $r" "typeof(r)=$(typeof(r))"
    r, logpdf_with_trans(dist, r, istrans(vi, vn))
end

function assume(spl::Sampler{<:Hamiltonian}, dists::Vector{<:Distribution}, vn::VarName, var::Any, vi::AbstractVarInfo)
    @assert length(dists) == 1 "[observe] Turing only support vectorizing i.i.d distribution"
    dist = dists[1]
    n = size(var)[end]

    vns = map(i -> copybyindex(vn, "[$i]"), 1:n)

    rs = vi[vns]  # NOTE: inside Turing the Julia conversion should be sticked to

    # acclogp!(vi, sum(logpdf_with_trans(dist, rs, istrans(vi, vns[1]))))

    if isa(dist, UnivariateDistribution) || isa(dist, MatrixDistribution)
        @assert size(var) == size(rs) "Turing.assume variable and random number dimension unmatched"
        var = rs
    elseif isa(dist, MultivariateDistribution)
        if isa(var, Vector)
            @assert length(var) == size(rs)[2] "Turing.assume variable and random number dimension unmatched"
            for i = 1:n
                var[i] = rs[:,i]
            end
        elseif isa(var, Matrix)
            @assert size(var) == size(rs) "Turing.assume variable and random number dimension unmatched"
            var = rs
        else
            error("[Turing] unsupported variable container")
        end
    end

    var, sum(logpdf_with_trans(dist, rs, istrans(vi, vns[1])))
end

observe(spl::Sampler{<:Hamiltonian}, d::Distribution, value::Any, vi::AbstractVarInfo) =
    observe(nothing, d, value, vi)

observe(spl::Sampler{<:Hamiltonian}, ds::Vector{<:Distribution}, value::Any, vi::AbstractVarInfo) =
observe(nothing, ds, value, vi)
