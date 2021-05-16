
struct PMLCoordinates{ZETA_DIM,T<:Number}
    "Represents the through-the-PML coordinate"
    ν::T
    "Represents the across-the-PML coordinate, for a PML in N dimensions ζ has dimension N-1"
    ζ::SVector{ZETA_DIM,T}
    """
    """
    function PMLCoordinates(ν, ζ)
        ζ_vec = SVector(ζ)
        new{length(ζ_vec), promote_type(typeof(ν),eltype(ζ_vec))}(ν, ζ_vec)
    end
end
