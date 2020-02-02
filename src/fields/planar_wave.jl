
struct PlanarWave{N,T} <: AbstractFieldFunction
    k::SVector{N,T}
    a::T
end

planarwave(k,a=one(first(k))) = PlanarWave(k,a)

(planarwave::PlanarWave)(x) = planarwave.a *exp(im*dot(x, planarwave.k))
