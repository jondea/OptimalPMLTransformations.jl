
abstract type AbstractFieldFunction <: Function end

struct FieldAndDerivativesAtPoint{T<:Real}
    u::Complex{T}
    du_dtν::Complex{T}
    du_dζ::Complex{T}
    d2u_dtν2::Complex{T}
    d2u_dtνdζ::Complex{T}
    d3u_dtν3::Complex{T}
end

# Access functions
u(field::FieldAndDerivativesAtPoint) = field.u
du_dtν(field::FieldAndDerivativesAtPoint) = field.du_dtν
du_dζ(field::FieldAndDerivativesAtPoint) = field.du_dζ
d2u_dtν2(field::FieldAndDerivativesAtPoint) = field.d2u_dtν2
d2u_dtνdζ(field::FieldAndDerivativesAtPoint) = field.d2u_dtνdζ
d3u_dtν3(field::FieldAndDerivativesAtPoint) = field.d3u_dtν3

import Base.(+)
function +(f1::FieldAndDerivativesAtPoint{T}, f2::FieldAndDerivativesAtPoint{T}) where {T}
    FieldAndDerivativesAtPoint{T}(f1.u+f2.u, f1.du_dtν+f2.du_dtν, f1.du_dζ+f2.du_dζ, f1.d2u_dtν2+f2.d2u_dtν2, f1.d2u_dtνdζ+f2.d2u_dtνdζ, f1.d3u_dtν3+f2.d3u_dtν3)
end

import Base.(*)
function *(a::Number, f::FieldAndDerivativesAtPoint{T}) where {T}
    FieldAndDerivativesAtPoint{T}(a*f.u, a*f.du_dtν, a*f.du_dζ, a*f.d2u_dtν2, a*f.d2u_dtνdζ, a*f.d3u_dtν3)
end

dtν_dν(u::FieldAndDerivativesAtPoint, U::FieldAndDerivativesAtPoint)::Complex = - U.u / u.du_dtν
dtν_dζ(u::FieldAndDerivativesAtPoint, U::FieldAndDerivativesAtPoint, ν)::Complex = (U.du_dζ*(1-ν) - u.du_dζ) / u.du_dtν

include("planar_wave.jl")

include("planar_wave_series.jl")

include("single_angular_fourier_mode.jl")

include("hankel_series.jl")
