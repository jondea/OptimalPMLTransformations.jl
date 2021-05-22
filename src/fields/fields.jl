
abstract type AbstractFieldFunction <: Function end

struct FieldAndDerivativesAtPoint{T<:Real}
    u::Complex{T}
    ∂u_∂tν::Complex{T}
    ∂u_∂tζ::Complex{T}
    ∂2u_∂tν2::Complex{T}
    ∂2u_∂tν∂ζ::Complex{T}
    ∂3u_∂tν3::Complex{T}
end

# # Access functions
# u(field::FieldAndDerivativesAtPoint) = field.u
# ∂u_∂tν(field::FieldAndDerivativesAtPoint) = field.∂u_∂tν
# ∂u_∂tζ(field::FieldAndDerivativesAtPoint) = field.∂u_∂tζ
# ∂2u_∂tν2(field::FieldAndDerivativesAtPoint) = field.∂2u_∂tν2
# ∂2u_∂tν∂ζ(field::FieldAndDerivativesAtPoint) = field.∂2u_∂tν∂ζ
# ∂3u_∂tν3(field::FieldAndDerivativesAtPoint) = field.∂3u_∂tν3

# import Base.(+)
# function +(f1::FieldAndDerivativesAtPoint{T}, f2::FieldAndDerivativesAtPoint{T}) where {T}
#     FieldAndDerivativesAtPoint{T}(f1.u+f2.u, f1.∂u_∂tν+f2.∂u_∂tν, f1.∂u_∂tζ+f2.∂u_∂tζ, f1.∂2u_∂tν2+f2.∂2u_∂tν2, f1.∂2u_∂tν∂ζ+f2.∂2u_∂tν∂ζ, f1.∂3u_∂tν3+f2.∂3u_∂tν3)
# end

# import Base.(*)
# function *(a::Number, f::FieldAndDerivativesAtPoint{T}) where {T}
#     FieldAndDerivativesAtPoint{T}(a*f.u, a*f.∂u_∂tν, a*f.∂u_∂tζ, a*f.∂2u_∂tν2, a*f.∂2u_∂tν∂ζ, a*f.∂3u_∂tν3)
# end

∂tν_∂ν(u::NamedTuple, U::NamedTuple)::Complex = - U.u / u.∂u_∂tν
∂tν_∂ζ(u::NamedTuple, U::NamedTuple, ν::Number)::Complex = (U.∂u_∂tζ*(1-ν) - u.∂u_∂tζ) / u.∂u_∂tν

include("planar_wave.jl")

include("planar_wave_series.jl")

include("single_angular_fourier_mode.jl")

include("hankel_series.jl")
