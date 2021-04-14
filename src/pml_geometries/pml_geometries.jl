
abstract type PMLGeometry <: Function end

abstract type AbstractFieldFunction <: Function end

struct PMLGeometryDerivatives{T<:Real}
    dx_dtν::SVector{Complex{T}}
    dx_dζ::SVector{Complex{T}}
    d2x_dtν2::SVector{Complex{T}}
    d2x_dtνdζ::SVector{Complex{T}}
    d3x_dtν3::SVector{Complex{T}}
end

include("pml_coordinates.jl")

include("annular_pml.jl")

include("x_aligned_pml.jl")

include("conformal_pml.jl")
