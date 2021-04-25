
struct CartesianCoordinates{T,N}
    x::SVector{T,N}
end

CartesianCoordinates(x...) = CartesianCoordinates(SVector(x...))

import Base: +, -, *, /, ==, ≈

+(a::CartesianCoordinates, b::CartesianCoordinates) = CartesianCoordinates(a.x+b.x)
-(a::CartesianCoordinates, b::CartesianCoordinates) = CartesianCoordinates(a.x-b.x)
*(a::Number, b::CartesianCoordinates) = CartesianCoordinates(a*b.x)
*(b::CartesianCoordinates, a::Number) = CartesianCoordinates(b.x*a)
/(b::CartesianCoordinates, a::Number) = CartesianCoordinates(b.x/a)
-(a::CartesianCoordinates) = CartesianCoordinates(-a.x)
≈(a::CartesianCoordinates, b::CartesianCoordinates) = a.x ≈ b.x
==(a::CartesianCoordinates, b::CartesianCoordinates) = a.x == b.x
