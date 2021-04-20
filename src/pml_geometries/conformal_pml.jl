
abstract type ConformalPML <: PMLGeometry end

"Should return vector (preferrably SVector) which represents inner boundary of PML as function of ζ::AbstractVector"
function inner_pml(::ConformalPML, ζ::Number)::SVector
    throw(MethodError("All subtypes of ConformalPML must implement inner_pml"))
end

"Should return vector (preferrably SVector) which represents PML direction as function of ζ::AbstractVector"
function pml_direction(::ConformalPML, ζ::Number)::SVector
    throw(MethodError("All subtypes of ConformalPML must implement pml_direction"))
end

"Should return vector (preferrably SVector) which represents inner boundary of PML as function of ζ::AbstractVector"
function inner_pml_deriv(::ConformalPML, ζ::Number)::SVector
    throw(MethodError("All subtypes of ConformalPML must implement inner_pml"))
end

"Should return vector (preferrably SVector) which represents PML direction as function of ζ::AbstractVector"
function pml_direction_deriv(::ConformalPML, ζ::Number)::SVector
    throw(MethodError("All subtypes of ConformalPML must implement pml_direction"))
end

"Should map from x to ζ"
function ζ_from_x(::ConformalPML, coords::SVector)::Number
    throw(MethodError("All subtypes of ConformalPML must implement pml_direction"))
end

function (geom::ConformalPML)(::PMLGeometryDerivatives, coords::PMLCoordinates)
    ν = c.ν; ζ = c.ζ
    X = inner_pml(pml, ζ)
    p = pml_direction(pml, ζ)
    dX_dζ = inner_pml_deriv(pml, ζ)
    dp_dζ = pml_direction_deriv(pml, ζ)
    dx_dν = p
    dx_dζ = dX_dζ + ν * dp_dζ
    d2x_dνdζ = dp_dζ
    d2x_dν2 = 0
    d3x_dν3 = 0
    PMLGeometryDerivatives(dx_dν, dx_dζ, d2x_dνdζ, d2x_dν2, d3x_dν3)
end

import Base.convert
function convert(::Val{PMLCoordinates}, pml::ConformalPML, c::CartesianCoordinates)
    x = c.x
    ζ = ζ_from_x(pml, x)
    X = inner_pml(pml, ζ)
    p = pml_direction(pml, ζ)
    ν = @einsum (x-X)[i]*p[i]
    PMLCoordinates{1}(ν, (ζ,))
end
function convert(::Val{CartesianCoordinates}, pml::ConformalPML, c::PMLCoordinates)
    ν = c.ν; ζ = c.ζ
    X = pml.X(ζ); p = pml.p(ζ)
    CartesianCoordinates{2}(X + ν*p)
end
