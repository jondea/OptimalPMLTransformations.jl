
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
