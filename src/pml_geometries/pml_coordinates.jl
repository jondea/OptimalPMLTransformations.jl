
struct PMLCoordinates{ZETA_DIM,T}
    ν::T # Represents the through-the-PML coordinate
    ζ::SVector{ZETA_DIM,T} # Represents the across-the-PML coordinate, for a PML in N dimensions ζ has dimension N-1
end

PMLCoordinates(ν, ζ) = PMLCoordinates(ν, SVector(ζ))
