

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

import Base.convert
convert(::Val{PMLCoordinates}, pml::AnnularPML, polar_coords::PolarCoordinates) = PMLCoordinates{1}((r-pml.R)/pml.δ, SVector(θ))
convert(::Val{PolarCoordinates}, pml::AnnularPML, polar_coords::PMLCoordinates) = SVector(pml.R+ν*pml.δ, ζ)
