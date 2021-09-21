
struct HankelSeries{KT, AT} <: AbstractFieldFunction
    k::KT
    a::AT
end

import Base: +
function +(u1::SingleAngularFourierMode, u2::SingleAngularFourierMode)
    if (u1.k != u2.k) error("wavenumbers must be the same when adding two fields") end
    HankelSeries(u1.k, Dict(u1.m=>u1.a, u2.m=>u2.a))
end

function pad(a::AbstractUnitRange, n::Integer)
    (minimum(a)-n):(maximum(a)+n)
end

function padded_hankelh1_vec(indices, z, padding)
    n_vec = pad(indices, padding)
    return OffsetVector(hankelh1.(n_vec, z), n_vec)
end

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

function (f::HankelSeries)(::Type{NamedTuple{(:u, :∂u_∂tr, :∂u_∂tθ, :∂2u_∂tr2, :∂2u_∂tr∂tθ, :∂3u_∂tr3)}}, p::PolarCoordinates)
    r = p.r
    θ = p.θ
    k = f.k
    a = f.a
    H = padded_hankelh1_vec(eachindex(a), k*r, 3)

    eⁱᶿ = exp(im*θ)
    ∑(fnc) = sum(fnc, eachindex(a))

    u        = ∑(n -> a[n] * (eⁱᶿ^n) *         H[n])
    ∂u_∂tr   = ∑(n -> a[n] * (eⁱᶿ^n) * k*     (H[n-1] - H[n+1])/2)
    ∂2u_∂tr2 = ∑(n -> a[n] * (eⁱᶿ^n) * k*k*   (H[n-2] - 2*H[n] + H[n+2])/4)
    ∂3u_∂tr3 = ∑(n -> a[n] * (eⁱᶿ^n) * k*k*k* (H[n-3] - 3*H[n-1] + 3*H[n+1] - H[n+3])/8)

    ∂u_∂tθ     = ∑(n -> a[n] * im*n*(eⁱᶿ^n) *     H[n])
    ∂2u_∂tr∂tθ = ∑(n -> a[n] * im*n*(eⁱᶿ^n) * k* (H[n-1] - H[n+1])/2)

    return (;u, ∂u_∂tr, ∂u_∂tθ, ∂2u_∂tr2, ∂2u_∂tr∂tθ, ∂3u_∂tr3)
end


function ∂u_∂tr(f::HankelSeries, p::PolarCoordinates)
    r = p.r
    θ = p.θ
    k = f.k
    a = f.a
    H = padded_hankelh1_vec(eachindex(a), k*r, 1)

    eⁱᶿ = exp(im*θ)
    ∑(fnc) = sum(fnc, eachindex(a))

    return ∑(n -> a[n] * (eⁱᶿ^n) * k*     (H[n-1] - H[n+1])/2)
end

function (f::HankelSeries)(::Type{NamedTuple{(:u, :∂u_∂tr)}}, p::PolarCoordinates)
    r = p.r
    θ = p.θ
    k = f.k
    a = f.a
    n_vec = (minimum(eachindex(a))-1):(maximum(eachindex(a))+1)
    H = OffsetVector(hankelh1.(n_vec, k*r), n_vec)

    eⁱᶿ = exp(im*θ)
    ∑(fnc) = sum(fnc, eachindex(a))

    u        = ∑(n -> a[n] * (eⁱᶿ^n) *         H[n])
    ∂u_∂tr   = ∑(n -> a[n] * (eⁱᶿ^n) * k*     (H[n-1] - H[n+1])/2)

    return (;u, ∂u_∂tr)
end

function (f::HankelSeries)(::Type{NamedTuple{(:u, :∂u_∂tν, :∂u_∂tζ, :∂2u_∂tν2, :∂2u_∂tν∂tζ, :∂3u_∂tν3)}}, p::PMLCoordinates, pml::AnnularPML)

    polar_coords = convert(PolarCoordinates, p, pml)

    (u, ∂u_∂tr, ∂u_∂tθ, ∂2u_∂tr2, ∂2u_∂tr∂tθ, ∂3u_∂tr3)  = f(NamedTuple{(:u, :∂u_∂tr, :∂u_∂tθ, :∂2u_∂tr2, :∂2u_∂tr∂tθ, :∂3u_∂tr3)}, polar_coords)

    ∂tr_∂tν = pml.δ

    ∂u_∂tν = ∂u_∂tr*∂tr_∂tν
    ∂u_∂tζ = ∂u_∂tθ
    ∂2u_∂tν2 = ∂2u_∂tr2*∂tr_∂tν^2
    ∂2u_∂tν∂tζ = ∂2u_∂tr∂tθ*∂tr_∂tν
    ∂3u_∂tν3 = ∂3u_∂tr3*∂tr_∂tν^3
    return (;u, ∂u_∂tν, ∂u_∂tζ, ∂2u_∂tν2, ∂2u_∂tν∂tζ, ∂3u_∂tν3)
end

diffbesselj(ν, z, h=besselj(ν, z), hm1=besselj(ν-1, z)) = hm1 - ν/z*h

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
