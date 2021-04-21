
struct SingleAngularFourierMode{N,T} <: AbstractFieldFunction
    k::T
    m::Int
    a::T
end

single_hankel_mode(k,m,a=one(k)) = SingleAngularFourierMode(k,m,a)
single_angular_fourier_mode(k,m,a=one(k)) = SingleAngularFourierMode(k,m,a)

(f::SingleAngularFourierMode)(p::PolarCoordinates) = f.a *exp(im*m*p.θ) * hankel1(f.m, f.k*p.r)
(f::SingleAngularFourierMode)(p::PMLCoordinates, g::PMLGeometry) = f(convert(PolarCoordinates, p, g))


# General case, can we do something with Jacobians?
function (f::SingleAngularFourierMode, p::PolarCoordinates, g::AnnularPML, ::FieldAndDerivativesAtPoint)

    u
    du_dtν
    du_dζ
    d2u_dtν2
    d2u_dtνdζ
    d3u_dtν3

    return FieldAndDerivativesAtPoint{T}(u, du_dtν, du_dζ, d2u_dtν2, d2u_dtνdζ, d3u_dtν3)
end
