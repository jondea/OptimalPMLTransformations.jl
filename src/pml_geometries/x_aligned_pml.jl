struct XAlignedRectangularPML{T} <: PMLGeometry
    X::T # x value at which the PML begins
    δ::T # PML thickness
end

# x = X + δ*tν
# y = ζ

function (geom::XAlignedRectangularPML)(::PMLGeometryDerivatives, coords::PMLCoordinates)
    dx_dtν = (δ, 0)
    dx_dζ = (0, 1)
    d2x_dtνdζ = (0, 0)
    d2x_dtν2 = (0, 0)
    d3x_dtν3 = (0, 0)
    PMLGeometryDerivatives(dx_dtν, dx_dζ, d2x_dtνdζ, d2x_dtν2, d3x_dtν3)
end

import Base.convert
function convert(::Val{PMLCoordinates}, pml::XAlignedRectangularPML, c::CartesianCoordinates)
    x = c.x[1]; y = c.x[2]
    X = pml.X; δ = pml.δ
    PMLCoordinates{1}((x - X)/δ, (y,))
end
function convert(::Val{CartesianCoordinates}, pml::XAlignedRectangularPML, c::PMLCoordinates)
    tν = c.tν; ζ = c.ζ
    X = pml.X; δ = pml.δ
    CartesianCoordinates{2}(X + δ*tν, ζ[1])
end
