struct XAlignedRectangularPML{T} <: PMLGeometry
    X::T # x value at which the PML begins
    δ::T # PML thickness
end

cartesian_to_pml_coordinates(pml::XAlignedRectangularPML, x::SVector) = PMLCoordinates((x[1]-pml.X)/pml.δ, SVector(x[2:end]...))
pml_to_cartesian_coordinates(pml::XAlignedRectangularPML, ν, ζ) = SVector(pml.X+pml.δ*ν, ζ[2:end]...)