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

import Base: convert
function Base.convert(::Type{PMLCoordinates}, c::CartesianCoordinates, pml::XAlignedRectangularPML)
    x = c.x[1]; y = c.x[2]
    X = pml.X; δ = pml.δ
    PMLCoordinates{1}(x - X, (y,))
end
function Base.convert(::Type{CartesianCoordinates}, c::PMLCoordinates, pml::XAlignedRectangularPML)
    ν = c.ν; ζ = c.ζ
    X = pml.X; δ = pml.δ
    CartesianCoordinates(X + ν, ζ[1])
end

tx_tν_jacobian(pml::XAlignedRectangularPML) = I
tν_ν_jacobian(::XAlignedRectangularPML, ∂tν_∂ν) = SDiagonal(∂tν_∂ν, 1)
ν_x_jacobian(pml::XAlignedRectangularPML) = I
