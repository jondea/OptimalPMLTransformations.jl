
struct HankelSeries{KT, AT} <: AbstractFieldFunction
    k::KT
    a::AT
end

import Base: +
function +(u1::SingleAngularFourierMode, u2::SingleAngularFourierMode)
    if (u1.k != u2.k) error("wavenumbers must be the same when adding two fields") end
    HankelSeries(u1.k, Dict(u1.m=>u1.a, u2.m=>u2.a))
end

single_mode_contrib(r,θ,m,k,a) = a *exp(im*m*θ) * hankel1(m, k*r)

function (f::HankelSeries)(p::PolarCoordinates)
    r = p.r
    θ = p.θ
    k = f.k
    h = zero(k)
    for m in eachindex(f.a)
        @inbounds a = f.a[m]
        h += a * exp(im*m*θ) * hankelh1(m, k*r)
    end
    return h
end

function (f::HankelSeries, r, θ, ::FieldAndDerivativesAtPoint)
    h = zero(f.k)
    for (m, a) in zip(eachindex(f.a), f.a)
        h += a *exp(im*m*θ) * hankelh1(m, f.k*r)
    end
    return h
end

scattered_coef(n, k; A=1.0) = (1.0*im)^n*diffbesselj(n,k*A)/diffhankelh1(n,k*A)
scattered_coef(ind::AbstractArray, k; A=1.0) = map(n->scattered_coef(n,k;A=A), Base.Slice(ind))

scattered_coef_k_0(n) = im*(1.0*im)^n
scattered_coef_k_0(ind::AbstractArray) = map(scattered_coef_k_0, Base.Slice(ind))

"Coefficients where there is a pole at z=z_pole and θ=0"
function two_mode_pole_coef(n, z_pole, θ_pole=0.0)
    if n == 0
        1.0 + 0.0im
    elseif n == 1
        -diffhankelh1(0, z_pole)/(diffhankelh1(1, z_pole)*exp(im*θ_pole))
    else
        0.0 + 0.0im
    end
end
two_mode_pole_coef(ind::AbstractArray, z_pole, θ_pole=0.0) = map(n->two_mode_pole_coef(n, z_pole, θ_pole), Base.Slice(ind))
