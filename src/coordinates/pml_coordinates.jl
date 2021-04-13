
struct PMLCoordinates{ZETA_DIM,T}
    "Represents the through-the-PML coordinate"
    ν::T
    "Represents the across-the-PML coordinate, for a PML in N dimensions ζ has dimension N-1"
    ζ::SVector{ZETA_DIM,T}
end

PMLCoordinates(ν, ζ) = PMLCoordinates(ν, SVector(ζ))
