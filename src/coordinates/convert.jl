
function convert(::Val{CartesianCoordinates}, polar::PolarCoordinates)
    r = polar.r
    θ = polar.θ
    CartesianCoordinates(r*cos(θ), r*sin(θ))
end

function convert(::Val{PolarCoordinates}, c::CartesianCoordinates)
    x = c.x[1]
    y = c.x[2]
    PolarCoordinates(sqrt(x^2+y^2), atan(y,x))
end

# Implement Jacobians
