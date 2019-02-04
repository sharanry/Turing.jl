function Sample(vi::UntypedVarInfo)
    values = Dict{Symbol, Dict{Symbol, Real}}()
    for vn in keys(vi)
        values[s][sym_idx(vn)] = vi[vn]
    end
    values = Dict(k => copy.(values[k]) for k in keys(values))
    nt = (;values..., lp = getlogp(vi))
    Sample(0.0, nt)
end
@generated function Sample(vi::TypedVarInfo{Tvis}) where Tvis
    nt_args = []
    for f in fieldnames(Tvis)
        push!(expr.args, quote
            value_$f = Dict{Symbol, eltype(vi.vis.$f.vals)}()
            for vn in keys(vi.vis.$f.idcs)
                value_$f[sym_idx(vn)] = vi[vn]
            end
        end)
        push!(nt_args, :($f = value_$f))
    end
    push!(nt_args, :(lp = getlogp(vi)))
    end)
    if length(nt_args) == 0
        return quote
            nt = NamedTuple()
            return Sample(0.0, nt)
        end
    else
        return quote
            nt = ($(nt_args...),)
            return Sample(0.0, nt)
        end
    end
end

# VarInfo, combined with spl.info, to Sample
function Sample(vi::AbstractVarInfo, spl::Sampler)
    s = Sample(vi)

    if isdefined(spl.info, :wum)
        s.info.epsilon = getss(spl.info.wum)
    end

    if isdefined(spl.info, :lf_num)
        s.info.lf_num = spl.info.lf_num
    end

    if isdefined(spl.info, :eval_num)
        s.info.eval_num = spl.info.eval_num
    end

    return s
end

using InteractiveUtils

function sample(model, alg; kwargs...)
    spl = get_sampler(model, alg; kwargs...)
    samples = init_samples(alg; kwargs...)
    vi = init_varinfo(model, spl; kwargs...)
    @code_warntype _sample(vi, samples, spl, model, alg, CHUNKSIZE[], false, nothing, 0, STAN_DEFAULT_ADAPT_CONF)
    _sample(vi, samples, spl, model, alg)
end

function init_samples(alg; kwargs...)
    n = get_sample_n(alg; kwargs...)
    weight = 1 / n
    samples = init_samples(n, weight)
    return samples
end

function get_sample_n(alg; reuse_spl_n = 0, kwargs...)
    if reuse_spl_n > 0
        return reuse_spl_n
    else
        alg.n_iters
    end
end

function init_samples(sample_n, weight)
    samples = Array{Sample}(undef, sample_n)
    weight = 1 / sample_n
    for i = 1:sample_n
        samples[i] = Sample(weight)
    end
    return samples
end

function get_sampler(model, alg; kwargs...)
    spl = default_sampler(model, alg; kwargs...)
    if alg isa AbstractGibbs
        @assert typeof(spl.alg) == typeof(alg) "[Turing] alg type mismatch; please use resume() to re-use spl"
    end
    return spl
end

function default_sampler(model, alg; reuse_spl_n = 0, resume_from = nothing, kwargs...)
    if reuse_spl_n > 0
        return resume_from.info[:spl]
    else
        return Sampler(alg, model)
    end
end

function init_varinfo(model, spl; kwargs...)
    vi = TypedVarInfo(default_varinfo(model, spl; kwargs...))
    return vi
end

function default_varinfo(model, spl; resume_from = nothing, kwargs...)
    if resume_from == nothing
        vi = VarInfo()
        model(vi, HamiltonianRobustInit())
        return vi
    else
        return resume_from.info[:vi]
    end
end
