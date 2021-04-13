
struct CartesianCoordinate{T,N}
    x::SVector{T,N}
end

CartesianCoordinate(x) = CartesianCoordinate(SVector(x))
