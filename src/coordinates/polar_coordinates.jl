
struct PolarCoordinates{T}
    "Coordinate in radial direction"
    r::T
    "Coordinate in angular direction"
    θ::SVector{T}
end

PMLCoordinates(ν, ζ) = PMLCoordinates(ν, SVector(ζ))

