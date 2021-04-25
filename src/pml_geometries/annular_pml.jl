

"""
    AnnularPML{T}(R::T, δ::T) <: PMLGeometry

Represents a annular PML with a radius `R` and a thickness `δ`.
"""
struct AnnularPML{T} <: PMLGeometry
    "Inner radius of PML"
    R::T
    "PML thickness"
    δ::T
end

function (geom::AnnularPML)(::PMLGeometryDerivatives, coords::PMLCoordinates)
    dx_dtν = (δ, 0)
    dx_dζ = (0, 1)
    d2x_dtνdζ = (0, 0)
    d2x_dtν2 = (0, 0)
    d3x_dtν3 = (0, 0)
    PMLGeometryDerivatives(dx_dtν, dx_dζ, d2x_dtνdζ, d2x_dtν2, d3x_dtν3)
end

import Base: convert
function Base.convert(::Type{PMLCoordinates}, c::PolarCoordinates, pml::AnnularPML)
    x = c.x[1]; y = c.x[2]
    X = pml.X; δ = pml.δ
    PMLCoordinates{1}((r-R)/δ, (θ,))
end
function Base.convert(::Type{PolarCoordinates}, c::PMLCoordinates, pml::AnnularPML)
    ν = c.ν; ζ = c.ζ
    X = pml.X; δ = pml.δ
    PolarCoordinates{2}(R + δ*ν, ζ[1])
end
