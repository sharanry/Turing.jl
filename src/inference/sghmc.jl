"""
    SGHMC(n_iters::Int, learning_rate::Float64, momentum_decay::Float64)

Stochastic Gradient Hamiltonian Monte Carlo sampler.

Usage:

```julia
SGHMC(1000, 0.01, 0.1)
```

Arguments:

- `n_iters::Int` : Number of samples to pull.
- `learning_rate::Float64` : The learning rate.
- `momentum_decay::Float64` : Momentum decay variable.

Example:

```julia
@model example begin
  ...
end

sample(example, SGHMC(1000, 0.01, 0.1))
```
"""
mutable struct SGHMC{AD, space} <: StaticHamiltonian{AD}
    n_iters::Int       # number of samples
    learning_rate::Float64   # learning rate
    momentum_decay::Float64   # momentum decay
    gid::Int
end
SGHMC(args...) = SGHMC{ADBackend()}(args...)
function SGHMC{AD}(learning_rate::Float64, momentum_decay::Float64, space...) where AD
    return SGHMC{AD, space}(1, learning_rate, momentum_decay, 0)
end
function SGHMC{AD}(n_iters, learning_rate, momentum_decay) where AD
    return SGHMC{AD, ()}(n_iters, learning_rate, momentum_decay, 0)
end
function SGHMC{AD}(n_iters, learning_rate, momentum_decay, space...) where AD
    return SGHMC{AD, space}(n_iters, learning_rate, momentum_decay, 0)
end
function SGHMC{AD1}(alg::SGHMC{AD2, space}, new_gid::Int) where {AD1, AD2, space}
    return SGHMC{AD1, space}(alg.n_iters, alg.learning_rate, alg.momentum_decay, new_gid)
end
function SGHMC{AD, space}(alg::SGHMC, new_gid::Int) where {AD, space}
    return SGHMC{AD, space}(alg.n_iters, alg.learning_rate, alg.momentum_decay, new_gid)
end

getspace(::SGHMC{<:Any, space}) where space = space

function step(model, spl::Sampler{<:SGHMC}, vi::AbstractVarInfo, is_first::Val{true})
    spl.alg.gid != 0 && link!(vi, spl)

    # Initialize velocity
    v = zeros(Float64, size(vi[spl]))
    spl.info[:v] = v

    spl.alg.gid != 0 && invlink!(vi, spl)
    return vi, true
end

function step(model, spl::Sampler{<:SGHMC}, vi::AbstractVarInfo, is_first::Val{false})
    # Set parameters
    η, α = spl.alg.learning_rate, spl.alg.momentum_decay

    Turing.DEBUG && @debug "X-> R..."
    if spl.alg.gid != 0
        link!(vi, spl)
        runmodel!(model, vi, spl)
    end

    Turing.DEBUG && @debug "recording old variables..."
    θ, v = vi[spl], spl.info[:v]
    _, grad = gradient_logp(θ, vi, model, spl)
    verifygrad(grad)

    # Implements the update equations from (15) of Chen et al. (2014).
    Turing.DEBUG && @debug "update latent variables and velocity..."
    θ .+= v
    v .= (1 - α) .* v .+ η .* grad .+ rand.(Normal.(zeros(length(θ)), sqrt(2 * η * α)))

    Turing.DEBUG && @debug "saving new latent variables..."
    vi[spl] = θ

    Turing.DEBUG && @debug "R -> X..."
    spl.alg.gid != 0 && invlink!(vi, spl)

    Turing.DEBUG && @debug "always accept..."
    return vi, true
end
