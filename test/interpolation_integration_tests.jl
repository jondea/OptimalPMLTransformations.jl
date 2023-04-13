using OptimalPMLTransformations
# import OptimalPMLTransformations: integrate_between, InterpLine, InterpPoint
using Tau

import Test: @test, @testset

# @testset "Integration"
begin

    R = 2.0
    δ = 1.0
    pml = AnnularPML(R, δ)
    k = 1.0
    a = two_mode_pole_coef(0:1, 2.0+1.0im)
    u = HankelSeries(k, a)

    θ₋ = -τ/8
    θ₊ = τ/8

    u_fnc_νζ(tν, ζ) = u(NamedTuple{(:u, :∂u_∂tν, :∂u_∂tζ, :∂2u_∂tν2, :∂2u_∂tν∂tζ, :∂3u_∂tν3)}, PMLCoordinates(tν,ζ), pml)

    intrp = interpolation(u, pml, range(θ₋, θ₊, length=11), 0.999; ε=1e-8)

    integrand(ν::Number, ∂tν_∂ν::Number) = 1.0/∂tν_∂ν + ((1-ν)^2)*∂tν_∂ν
    integrand(p::InterpPoint) = integrand(p.ν, p.∂tν_∂ν)

    int_order = 2
    region1 = intrp.continuous_region[1]

    integral_i = let
        n_refine = 4
        region = deepcopy(intrp.continuous_region[1])
        for _ in 1:4 refine_in_ζ!(region, u, pml) end
        for line in region.lines
            for _ in 1:n_refine refine!(line, u, pml) end
        end
        integrate(region, integrand; order=int_order)
    end
    @show integral_i


    integrand_patch_fnc(patch, ζ0, ζ1, ν, ζ) = integrand(evaluate_and_correct(u, pml, patch, ζ0, ζ1, ν, ζ))
    integral_h = integrate_hcubature(intrp, integrand_patch_fnc; atol=1e-14, rtol=1e-12, maxevals=1000_000)

    @show integral_h

    integral_q = integrate_quad(u, pml, integrand, (θ₋, θ₊), 1000)

    @show integral_q

    @test integral_q ≈ integral_i
    @test integral_i ≈ integral_h
    @test integral_h ≈ integral_q

end
