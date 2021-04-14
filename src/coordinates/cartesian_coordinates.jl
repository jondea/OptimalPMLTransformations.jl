
struct CartesianCoordinates{T,N}
    x::SVector{T,N}
end

CartesianCoordinates(x) = CartesianCoordinates(SVector(x))
