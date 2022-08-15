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

# Note, we can set the default range to [0,τ] because an Annular PML is a full turn.
# ν_max=1 means we attempt to set the whole PML region
function add_interpolation!(p::InvHankelSeriesPML; ζs = range(0.0, τ, length=11), ν_max=1.0, kwargs...)
    p.interp = interpolate(p.u, p.pml, ζs, ν_max; kwargs...)
end

has_interpolation(p::InvHankelSeriesPML) = !isempty(p.interp.continuous_region)
