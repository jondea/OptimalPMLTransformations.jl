
struct HankelSeries{N,T} <: AbstractFieldFunction
    k::T
    a::OffsetVector{T}
end

single_mode_contrib(r,θ,m,k,a) = a *exp(im*m*θ) * hankel1(m, k*r)

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
