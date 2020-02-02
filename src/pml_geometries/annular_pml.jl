

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

polar_to_pml_coordinates(pml::AnnularPML, r, θ) = PMLCoordinates{1}((r-pml.R)/pml.δ, SVector(θ))
pml_to_polar_coordinates(pml::AnnularPML, ν, ζ) = SVector(pml.R+ν*pml.δ, ζ)
