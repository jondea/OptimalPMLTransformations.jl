
struct SingleAngularFourierMode{N,T} <: AbstractFieldFunction
    k::T
    m::Int
    a::T
end

single_hankel_mode(k,m,a=one(first(k))) = SingleAngularFourierMode(k,a)
single_angular_fourier_mode(k,m,a=one(first(k))) = SingleAngularFourierMode(k,a)

(f::SingleAngularFourierMode)(r, θ) = f.a *exp(im*m*θ) * hankel1(f.m, f.k*r)
