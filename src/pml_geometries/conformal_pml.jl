
struct ConformalPML <: PMLGeometry
    X::Function # Should return vector (preferrably SVector) which represents inner boundary of PML as function of ζ::AbstractVector
    p::Function # Should return vector (preferrably SVector) which represents PML direction as function of ζ::AbstractVector
    x_to_ζ::Function # Should map from x to ζ
end

pml_to_cartesian_coordinates(pml::ConformalPML, ν, ζ) = pml.X(ζ) + pml.p(ζ) * ν
function cartesian_to_pml_coordinates(pml::ConformalPML, x::SVector)
    ζ = pml.x_to_ζ(x)
    ν = dot(x-pml.X(ζ), pml.p(ζ))
    PMLCoordinates(ν, ζ)
end
