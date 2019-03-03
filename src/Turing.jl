module Turing

##############
# Dependency #
########################################################################
# NOTE: when using anything from external packages,                    #
#       let's keep the practice of explictly writing Package.something #
#       to indicate that's not implemented inside Turing.jl            #
########################################################################

using Requires, Reexport, ForwardDiff
using Bijectors, StatsFuns, SpecialFunctions
using Statistics, LinearAlgebra, ProgressMeter
using Markdown, Libtask, MacroTools
@reexport using Distributions, MCMCChain, Libtask
using Flux.Tracker: Tracker

import Base: ~, convert, promote_rule, rand, getindex, setindex!
import Distributions: sample
import MCMCChain: AbstractChains, Chains

const PROGRESS = Ref(true)
function turnprogress(switch::Bool)
    @info("[Turing]: global PROGRESS is set as $switch")
    PROGRESS[] = switch
end

# Constants for caching
const CACHERESET  = 0b00
const CACHEIDCS   = 0b10
const CACHERANGES = 0b01

const DEBUG = Bool(parse(Int, get(ENV, "DEBUG_TURING", "0")))

"""
    struct Model{pvars, dvars, F, TData, TDefaults}
        f::F
        data::TData
        defaults::TDefaults
    end
    
A `Model` struct with parameter variables `pvars`, data variables `dvars`, inner 
function `f`, `data::NamedTuple` and `defaults::NamedTuple`.
"""
struct Model{pvars, dvars, F, TData, TDefaults}
    f::F
    data::TData
    defaults::TDefaults
end
function Model{pvars, dvars}(f::F, data::TD, defaults::TDefaults) where {pvars, dvars, F, TD, TDefaults}
    return Model{pvars, dvars, F, TD, TDefaults}(f, data, defaults)
end
get_pvars(m::Model{params}) where {params} = Tuple(params.types)
get_dvars(m::Model{params, data}) where {params, data} = Tuple(data.types)
get_defaults(m::Model) = m.defaults
@generated function in_pvars(::Val{sym}, ::Model{params}) where {sym, params}
    return sym in params.types ? :(true) : :(false)
end
@generated function in_dvars(::Val{sym}, ::Model{params, data}) where {sym, params, data}
    return sym in data.types ? :(true) : :(false)
end
function Base.getproperty(m::Model, f::Symbol)
    f === :pvars && return get_pvars(m)
    return getfield(m, f)
end

(model::Model)(args...; kwargs...) = model.f(args..., model; kwargs...)
function runmodel! end

abstract type AbstractSampler end
"""
    Sampler{T}

Generic interface for implementing inference algorithms.
An implementation of an algorithm should include the following:

1. A type specifying the algorithm and its parameters, derived from InferenceAlgorithm
2. A method of `sample` function that produces results of inference, which is where actual inference happens.

Turing translates models to chunks that call the modelling functions at specified points. The dispatch is based on the value of a `sampler` variable. To include a new inference algorithm implements the requirements mentioned above in a separate file,
then include that file at the end of this one.
"""
mutable struct Sampler{T} <: AbstractSampler
    alg   ::  T
    info  ::  Dict{Symbol, Any}         # sampler infomation
end
Sampler(alg, model) = Sampler(alg)

getspace(spl::Sampler) = getspace(typeof(spl))
getspace(::Type{<:Sampler{A}}) where A = getspace(A)

include("utilities/Utilities.jl")
using .Utilities
include("core/Core.jl")
using .Core
include("inference/Inference.jl")  # inference algorithms
using .Inference

@init @require CmdStan="593b3428-ca2f-500c-ae53-031589ec8ddd" @eval begin
    @eval Utilities begin
        using ..Turing.CmdStan: CmdStan, Adapt, Hmc
        include("utilities/stan-interface.jl")
    end
    @eval Inference begin
        using ..Turing.CmdStan: CmdStan
        DEFAULT_ADAPT_CONF_TYPE = Union{DEFAULT_ADAPT_CONF_TYPE, CmdStan.Adapt}
        STAN_DEFAULT_ADAPT_CONF = CmdStan.Adapt()
        
        Sampler(alg::Hamiltonian) =  Sampler(alg, CmdStan.Adapt())
        function Sampler(alg::Hamiltonian, adapt_conf::CmdStan.Adapt)
            _sampler(alg::Hamiltonian, adapt_conf)
        end
        include("inference/adapt/stan.jl")
    end
end
@init @require LogDensityProblems="6fdf6af0-433a-55f7-b3ed-c6c6e0b8df7c" @eval Inference begin
    using ..Turing.LogDensityProblems: LogDensityProblems, AbstractLogDensityProblem, ValueGradient
    struct FunctionLogDensity{F} <: AbstractLogDensityProblem
        dimension::Int
        f::F
    end

    LogDensityProblems.dimension(ℓ::FunctionLogDensity) = ℓ.dimension

    LogDensityProblems.logdensity(::Type{ValueGradient}, ℓ::FunctionLogDensity, x) = ℓ.f(x)::ValueGradient
end
@init @require DynamicHMC="bbc10e6e-7c05-544b-b16e-64fede858acb" @eval Inference begin
    using ..Turing.DynamicHMC: DynamicHMC, NUTS_init_tune_mcmc
    include("inference/dynamichmc.jl")
end

###########
# Exports #
###########

# Turing essentials - modelling macros and inference algorithms
export  @model,                 # modelling
        @VarName, 
        
        MH,                     # classic sampling
        Gibbs,
        
        HMC,                    # Hamiltonian-like sampling 
        SGLD, 
        SGHMC, 
        HMCDA, 
        NUTS,
        DynamicNUTS,
        
        IS,                     # particle-based sampling
        SMC, 
        CSMC, 
        PG, 
        PIMH, 
        PMMH, 
        IPMCMC,

        sample,                 # inference 
        setchunksize, 
        resume,

        auto_tune_chunk_size!,  # helper
        setadbackend, 
        setadsafe, 

        turnprogress,           # debugging

        Flat, 
        FlatPos, 
        BinomialLogit, 
        VecBinomialLogit

end
