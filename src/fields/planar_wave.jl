
struct PlanarWave{N,T} <: AbstractFieldFunction
    k::SVector{N,T}
    a::T
end

planarwave(k,a=one(first(k))) = PlanarWave(SVector(k),a)

(planarwave::PlanarWave)(x) = planarwave.a *exp(im*dot(x, planarwave.k))

function (f::HankelSeries)(r, θ)
    h = zero(f.k)
    for (m, a) in zip(eachindex(f.a), f.a)
        h += a *exp(im*m*θ) * hankel1(m, f.k*r)
    end
    return h
end

function (f::HankelSeries, r, θ, ::FieldAndDerivativesAtPoint)
    h = zero(f.k)
    for (m, a) in zip(eachindex(f.a), f.a)
        h += a *exp(im*m*θ) * hankel1(m, f.k*r)
    end
    return h
end
