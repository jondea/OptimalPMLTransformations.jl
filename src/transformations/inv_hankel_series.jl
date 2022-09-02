"""
    InvHankelSeriesPML(geom <: AnnularPML, mode <: HankelSeries, interp)

An optimal transformation defined on an Annular PML which is the inverse of a
series of angular Fourier/radial Hankel modes. The PML transformation stores an
interpolation to allow for efficient evaluation anywhere in the PML.
"""
mutable struct InvHankelSeriesPML{P <: AnnularPML, S <: HankelSeries}
    "Geometry of the PML on which the transformation is defined"
    geom::P
    "A field defined as a series of Hankel functions which we will invert to find the transformation"
    u::S
    "Interpolation"
    interp::Interpolation
end

InvHankelSeriesPML(geom, u) = InvHankelSeriesPML(geom, u, Interpolation())

# Note, we can set the default range to [0,τ] because an Annular PML is a full turn.
# ν_max=1 means we attempt to set the whole PML region
function add_interpolation!(p::InvHankelSeriesPML; num_ζs=11, ζs = range(0.0, τ, length=num_ζs), ν_max=1.0, kwargs...)
    p.interp = interpolate(p.u, p.pml, ζs, ν_max; kwargs...)
end

has_interpolation(p::InvHankelSeriesPML) = !isempty(p.interp)

function tr_and_jacobian(pml::InvHankelSeriesPML, patch::InterpPatch, coords::PolarCoordinates)
    return tr_and_jacobian(pml, patch, convert(PMLCoordinates, coords, pml.geom))
end

function tr_and_jacobian(pml::InvHankelSeriesPML, patch::InterpPatch, coords::PMLCoordinates)
    return evalute_and_correct(pml.u, pml.geom, patch, coords.ν, coords.ζ)
end
