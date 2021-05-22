
struct SingleAngularFourierMode{KT<:Number, MT<:Integer, AT<:Number} <: AbstractFieldFunction
    k::KT
    m::MT
    a::AT
end

single_hankel_mode(k,m,a=one(k)) = SingleAngularFourierMode(k,m,a)
single_angular_fourier_mode(k,m,a=one(k)) = SingleAngularFourierMode(k,m,a)

import Base: *
*(a::Number, u::SingleAngularFourierMode) = SingleAngularFourierMode(u.k, u.m, a*u.a)

(f::SingleAngularFourierMode)(p::PolarCoordinates) = f.a *exp(im*f.m*p.θ) * hankelh1(f.m, f.k*p.r)
(f::SingleAngularFourierMode)(c::CartesianCoordinates) = (f::SingleAngularFourierMode)(convert(PolarCoordinates, c))
(f::SingleAngularFourierMode)(p::PMLCoordinates, g::PMLGeometry) = f(convert(PolarCoordinates, p, g))

# General case, can we do something with Jacobians?
function (f::SingleAngularFourierMode, p::PolarCoordinates, g::AnnularPML, ::FieldAndDerivativesAtPoint)

    u
    ∂u_∂tν
    ∂u_∂tζ
    ∂2u_∂tν2
    ∂2u_∂tνdζ
    ∂3u_∂tν3

    return FieldAndDerivativesAtPoint{T}(u, ∂u_∂tν, ∂u_∂tζ, ∂2u_∂tν2, d2u_dtνdζ, ∂3u_∂tν3)
end
