
struct CartesianCoordinates{N,T<:Number}
    x::SVector{N,T}
    function CartesianCoordinates(x...)
        x_svec = SVector(x...)
        new{length(x_svec), eltype(x_svec)}(x_svec)
    end
end


import Base: -, ==, ≈
-(a::CartesianCoordinates, b::CartesianCoordinates) = a.x-b.x
≈(a::CartesianCoordinates, b::CartesianCoordinates) = a.x ≈ b.x
==(a::CartesianCoordinates, b::CartesianCoordinates) = a.x == b.x
