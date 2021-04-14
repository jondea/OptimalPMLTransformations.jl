
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

function (geom::ConformalPML)(::PMLGeometryDerivatives, coords::PMLCoordinates)
    dx_dtν =
    dx_dζ =
    d2x_dtνdζ =
    d2x_dtν2 =
    d3x_dtν3 =
    PMLGeometryDerivatives(dx_dtν, dx_dζ, d2x_dtνdζ, d2x_dtν2, d3x_dtν3)
end

import Base.convert
function convert(::Val{PMLCoordinates}, pml::ConformalPML, c::CartesianCoordinates)
    x = c.x
    ζ = x_to_ζ(x)
    X = pml.X(ζ); p = pml.X(ζ)
    ν = @einsum (x-X)[i]*p[i]
    PMLCoordinates{1}(ν, (ζ,))
end
function convert(::Val{CartesianCoordinates}, pml::ConformalPML, c::PMLCoordinates)
    tν = c.tν; ζ = c.ζ
    X = pml.X(ζ); p = pml.X(ζ)
    CartesianCoordinates{2}(X + δ*tν*p)
end
