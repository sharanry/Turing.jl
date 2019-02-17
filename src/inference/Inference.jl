module Inference

using ..Core, ..Core.VarReplay, ..Utilities
using Distributions, Libtask, Bijectors
using ProgressMeter, LinearAlgebra, Setfield
using ..Turing: PROGRESS, CACHERESET, AbstractSampler
using ..Turing: Sampler, Model, runmodel!, get_pvars, get_dvars
using ..Turing: in_pvars, in_dvars, Turing
using StatsFuns: logsumexp
using Parameters: @unpack

import Distributions: sample
import ..Core: getchunksize, getADtype
import ..Turing: getspace
import ..Utilities: Sample

export  InferenceAlgorithm,
        Hamiltonian,
        AbstractGibbs,
        GibbsComponent,
        StaticHamiltonian,
        AdaptiveHamiltonian,
        HamiltonianRobustInit,
        SampleFromPrior,
        AnySampler,
        MH, 
        Gibbs,      # classic sampling
        HMC, 
        SGLD, 
        SGHMC, 
        HMCDA, 
        NUTS,       # Hamiltonian-like sampling
        DynamicNUTS,
        IS, 
        SMC, 
        CSMC, 
        PG, 
        PIMH, 
        PMMH, 
        IPMCMC,  # particle-based sampling
        getspace,
        assume,
        observe,
        step,
        WelfordVar,
        WelfordCovar,
        NaiveCovar,
        get_var,
        get_covar,
        add_sample!,
        reset!

#######################
# Sampler abstraction #
#######################
abstract type AbstractAdapter end
abstract type InferenceAlgorithm end
abstract type AbstractGibbs <: InferenceAlgorithm end
abstract type Hamiltonian{AD} <: InferenceAlgorithm end
abstract type StaticHamiltonian{AD} <: Hamiltonian{AD} end
abstract type AdaptiveHamiltonian{AD} <: Hamiltonian{AD} end

getchunksize(::T) where {T <: Hamiltonian} = getchunksize(T)
getchunksize(::Type{<:Hamiltonian{AD}}) where AD = getchunksize(AD)
getADtype(alg::Hamiltonian) = getADtype(typeof(alg))
getADtype(::Type{<:Hamiltonian{AD}}) where {AD} = AD

# mutable struct HMCState{T<:Real}
#     epsilon  :: T
#     std     :: Vector{T}
#     lf_num   :: Integer
#     eval_num :: Integer
# end
#
#  struct Sampler{TH<:Hamiltonian,TA<:AbstractAdapter} <: AbstractSampler
#    alg   :: TH
#    state :: HMCState
#    adapt :: TA
#  end

"""
Robust initialization method for model parameters in Hamiltonian samplers.
"""
struct HamiltonianRobustInit <: AbstractSampler end
struct SampleFromPrior <: AbstractSampler end

# This can be removed when all `spl=nothing` is replaced with
#   `spl=SampleFromPrior`
const AnySampler = Union{Nothing, AbstractSampler}

# Helper functions
include("adapt/adapt.jl")
include("support/hmc_core.jl")
include("support/util.jl")

# Concrete algorithm implementations.
include("hmcda.jl")
include("nuts.jl")
include("sghmc.jl")
include("sgld.jl")
include("hmc.jl")
include("mh.jl")
include("is.jl")
include("smc.jl")
include("pgibbs.jl")
include("pmmh.jl")
include("ipmcmc.jl")
include("gibbs.jl")

include("sample.jl")
include("fallbacks.jl")

getspace(alg::InferenceAlgorithm) = getspace(typeof(alg))
for A in (:IPMCMC, :IS, :MH, :PMMH, :SMC, :Gibbs, :PG)
    @eval getspace(::Type{<:$A{space}}) where space = space
end
for A in (:HMC, :SGHMC, :SGLD, :HMCDA, :NUTS)
    @eval getspace(::Type{<:$A{<:Any, space}}) where space = space
end

end # module
