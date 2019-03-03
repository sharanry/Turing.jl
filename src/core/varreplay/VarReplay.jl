module VarReplay

using ...Turing: Turing, CACHERESET, CACHEIDCS, CACHERANGES, Sampler, Model, getspace
using ...Utilities: vectorize, reconstruct, reconstruct!
using Bijectors: SimplexDistribution
using Distributions

import Base: string, isequal, ==, hash, getindex, setindex!, push!, show, isempty
import Turing: link, invlink

export  VarName, 
        AbstractVarInfo,
        VarInfo,
        NewVarInfo,
        UntypedVarInfo,
        TypedVarInfo,
        uid, 
        sym, 
        getlogp, 
        set_retained_vns_del_by_spl!, 
        resetlogp!, 
        is_flagged, 
        unset_flag!, 
        setgid!, 
        copybyindex, 
        setorder!, 
        updategid!, 
        acclogp!, 
        istrans, 
        link!, 
        invlink!, 
        setlogp!, 
        getranges, 
        getrange, 
        getvns, 
        getval

###########
# VarName #
###########
struct VarName{sym}
    csym      ::    Symbol        # symbol generated in compilation time
    indexing  ::    String        # indexing
    counter   ::    Int           # counter of same {csym, uid}
end
VarName(csym, sym, indexing, counter) = VarName{sym}(csym, indexing, counter)

function Base.getproperty(vn::VarName{sym}, f::Symbol) where {sym}
    return f === :sym ? sym : getfield(vn, f)
end

# NOTE: VarName should only be constructed by VarInfo internally due to the nature of the counter field.

uid(vn::VarName) = (vn.csym, vn.sym, vn.indexing, vn.counter)
Base.hash(vn::VarName) = hash(uid(vn))

isequal(x::VarName, y::VarName) = hash(uid(x)) == hash(uid(y))
==(x::VarName, y::VarName)      = isequal(x, y)

Base.string(vn::VarName) = "{$(vn.csym),$(vn.sym)$(vn.indexing)}:$(vn.counter)"
Base.string(vns::Vector{<:VarName}) = replace(string(map(vn -> string(vn), vns)), "String" => "")

sym(vn::VarName) = Symbol("$(vn.sym)$(vn.indexing)")  # simplified symbol

cuid(vn::VarName) = (vn.csym, vn.sym, vn.indexing)    # the uid which is only available at compile time

copybyindex(vn::VarName, indexing::String) = VarName(vn.csym, vn.sym, indexing, vn.counter)

include("varinfo.jl")
include("typed_varinfo.jl")

end
