
struct HankelSeries{KT, AT} <: AbstractFieldFunction
    k::KT
    a::AT
end

wavenumber(hs::HankelSeries) = hs.k

import Base: +
function +(u1::SingleAngularFourierMode, u2::SingleAngularFourierMode)
    if (u1.k != u2.k) error("wavenumbers must be the same when adding two fields") end
    HankelSeries(u1.k, Dict(u1.m=>u1.a, u2.m=>u2.a))
end

function padded_hankelh1_vec(indices, z, padding)
    n_vec = padded_range(indices, padding)

    pos_n_vec = max(n_vec.start,0,-n_vec.stop):max(-n_vec.start,0,n_vec.stop) # == minimum(abs,n_vec):maximum(abs,n_vec)
    pos_h = hankelh1_seq(pos_n_vec, z)
    # This is quite hot, could be smarter about this
    return OffsetVector(map(n->(n%2==0 ? one(n) : sign(n))*pos_h[abs(n)], n_vec), n_vec)
end

# Taken from https://github.com/JuliaMath/SpecialFunctions.jl/pull/91
# Boaz Blankrot (bblankrot) and modified
function hankelh1_seq(P::UnitRange, z)
    v = OffsetVector{typeof(complex(z))}(undef, P)
    Hₚ₋₂ = hankelh1(first(P)-1, z)
    Hₚ₋₁ = hankelh1(first(P), z)
    v[first(P)] = Hₚ₋₁
    for p in firstrest(P)[2]
        Hₚ = (2*(p-1)/z)*Hₚ₋₁ - Hₚ₋₂
        v[p] = Hₚ
        Hₚ₋₂ = Hₚ₋₁; Hₚ₋₁ = Hₚ
    end
    return v
end

@generated function eval_hankel(h, NT, p, order_valtype)

    tuple_symbols = collect(NT.parameters[1])
    order = order_valtype.parameters[1]

    deriv_expr = Dict(
        :u          => quote a_n_exp_iθn *               H[n] end,
        :∂u_∂tr     => quote a_n_exp_iθn * k*         (H[n-1] - H[n+1])/2 end,
        :∂2u_∂tr2   => quote a_n_exp_iθn * k*k*     (H[n-2] - 2*H[n] + H[n+2])/4 end,
        :∂3u_∂tr3   => quote a_n_exp_iθn * k*k*k* (H[n-3] - 3*H[n-1] + 3*H[n+1] - H[n+3])/8 end,
        :∂u_∂tθ     => quote im*n*a_n_exp_iθn *     H[n] end,
        :∂2u_∂tr∂tθ => quote im*n*a_n_exp_iθn * k* (H[n-1] - H[n+1])/2 end,
    )
    return quote
        r = p.r
        θ = p.θ
        k = h.k
        a = h.a
        if isnan(k*r)
            cnan = (1 + im)*typeof(k*r)(NaN)
            return (;$([:($s = cnan) for s in tuple_symbols]...))
        end

        H = padded_hankelh1_vec(eachindex(a), k*r, $order)
        $([:($s = zero(complex(k*r))) for s in tuple_symbols]...)
        u = zero(complex(k*r))
        @fastmath @inbounds for n in eachindex(h.a)
            # u += $(deriv_expr[:u])
            a_n_exp_iθn = a[n] * exp(im*θ*n)
            $([:($s += $(deriv_expr[s])) for s in tuple_symbols]...)
        end
        return (;$([s for s in tuple_symbols]...))
    end
end

padding_needed(::Type{NamedTuple{(:u,)}}) = Val{0}()
padding_needed(::Type{NamedTuple{(:u, :∂u_∂tθ)}}) = Val{0}()
padding_needed(::Type{NamedTuple{(:∂u_∂tr,)}}) = Val{1}()
padding_needed(::Type{NamedTuple{(:u, :∂u_∂tr)}}) = Val{1}()
padding_needed(::Type{NamedTuple{(:u, :∂u_∂tr, :∂u_∂tθ)}}) = Val{1}()
padding_needed(::Type{NamedTuple{(:u, :∂u_∂tr, :∂u_∂tθ, :∂2u_∂tr2)}}) = Val{2}()
padding_needed(::Type{NamedTuple{(:u, :∂u_∂tr, :∂u_∂tθ, :∂2u_∂tr2, :∂2u_∂tr∂tθ)}}) = Val{2}()
padding_needed(::Type{NamedTuple{(:u, :∂u_∂tr, :∂u_∂tθ, :∂2u_∂tr2, :∂2u_∂tr∂tθ, :∂3u_∂tr3)}}) = Val{3}()
padding_needed(::Type{NamedTuple{(:u, :∂u_∂tr, :∂u_∂tθ, :∂2u_∂tr2, :∂3u_∂tr3)}}) = Val{3}()

function (f::HankelSeries)(nt::Type{T}, p::PolarCoordinates) where T<:NamedTuple
    eval_hankel(f, T((;u=1, ∂u_∂tr=1, ∂u_∂tθ=1, ∂2u_∂tr2=1, ∂2u_∂tr∂tθ=1, ∂3u_∂tr3=1)), p, padding_needed(nt))
end

function (f::HankelSeries)(p::PolarCoordinates)
    f(NamedTuple{(:u,)}, p).u
end

function ∂u_∂tr(f::HankelSeries, p::PolarCoordinates)
    f(NamedTuple{(:∂u_∂tr,)}, p).∂u_∂tr
end

function (f::HankelSeries)(::Type{NamedTuple{(:u, :∂u_∂tν, :∂u_∂tζ, :∂2u_∂tν2, :∂2u_∂tν∂tζ, :∂3u_∂tν3)}}, p::PMLCoordinates, pml::AnnularPML)

    polar_coords = convert(PolarCoordinates, p, pml)

    (;u, ∂u_∂tr, ∂u_∂tθ, ∂2u_∂tr2, ∂2u_∂tr∂tθ, ∂3u_∂tr3)  = f(NamedTuple{(:u, :∂u_∂tr, :∂u_∂tθ, :∂2u_∂tr2, :∂2u_∂tr∂tθ, :∂3u_∂tr3)}, polar_coords)

    ∂tr_∂tν = pml.δ

    ∂u_∂tν = ∂u_∂tr*∂tr_∂tν
    ∂u_∂tζ = ∂u_∂tθ
    ∂2u_∂tν2 = ∂2u_∂tr2*∂tr_∂tν^2
    ∂2u_∂tν∂tζ = ∂2u_∂tr∂tθ*∂tr_∂tν
    ∂3u_∂tν3 = ∂3u_∂tr3*∂tr_∂tν^3
    return (;u, ∂u_∂tν, ∂u_∂tζ, ∂2u_∂tν2, ∂2u_∂tν∂tζ, ∂3u_∂tν3)
end

function (f::HankelSeries)(::Type{NamedTuple{(:u, :∂u_∂tζ)}}, p::PMLCoordinates, pml::AnnularPML)

    polar_coords = convert(PolarCoordinates, p, pml)

    (;u, ∂u_∂tθ)  = f(NamedTuple{(:u, :∂u_∂tθ)}, polar_coords)

    ∂u_∂tζ = ∂u_∂tθ
    return (;u, ∂u_∂tζ)
end

function (f::HankelSeries)(::Type{NamedTuple{(:u, :∂u_∂tν)}}, p::PMLCoordinates, pml::AnnularPML)

    polar_coords = convert(PolarCoordinates, p, pml)

    (;u, ∂u_∂tr)  = f(NamedTuple{(:u, :∂u_∂tr)}, polar_coords)

    ∂tr_∂tν = pml.δ

    ∂u_∂tν = ∂u_∂tr*∂tr_∂tν
    return (;u, ∂u_∂tν)
end

function (f::HankelSeries)(::Type{NamedTuple{(:u, :∂u_∂tν, :∂u_∂tζ, :∂2u_∂tν2, :∂2u_∂tν∂tζ)}}, p::PMLCoordinates, pml::AnnularPML)

    polar_coords = convert(PolarCoordinates, p, pml)

    (;u, ∂u_∂tr, ∂u_∂tθ, ∂2u_∂tr2, ∂2u_∂tr∂tθ)  = f(NamedTuple{(:u, :∂u_∂tr, :∂u_∂tθ, :∂2u_∂tr2, :∂2u_∂tr∂tθ)}, polar_coords)

    ∂tr_∂tν = pml.δ

    ∂u_∂tν = ∂u_∂tr*∂tr_∂tν
    ∂u_∂tζ = ∂u_∂tθ
    ∂2u_∂tν2 = ∂2u_∂tr2*∂tr_∂tν^2
    ∂2u_∂tν∂tζ = ∂2u_∂tr∂tθ*∂tr_∂tν
    return (;u, ∂u_∂tν, ∂u_∂tζ, ∂2u_∂tν2, ∂2u_∂tν∂tζ)
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

two_mode_pole_series(k, r_pole) = HankelSeries(k, OffsetVector(two_mode_pole_coef(0:1, k*r_pole),0:1))

HankelSeries(u::SingleAngularFourierMode) = HankelSeries(u.k, OffsetVector([u.a], u.m:u.m))
HankelSeries(u::HankelSeries) = u
