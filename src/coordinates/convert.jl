

import Base: convert
function Base.convert(::Type{CartesianCoordinates}, polar::PolarCoordinates)
    r = polar.r
    θ = polar.θ
    CartesianCoordinates(r*cos(θ), r*sin(θ))
end

import Base: convert
function Base.convert(::Type{PolarCoordinates}, c::CartesianCoordinates)
    x = c.x[1]
    y = c.x[2]
    PolarCoordinates(sqrt(x^2+y^2), atan(y,x))
end

# Implement Jacobians
