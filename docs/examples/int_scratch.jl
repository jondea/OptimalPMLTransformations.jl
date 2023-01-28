
using OptimalPMLTransformations
import OptimalPMLTransformations.evalute_and_correct
using Tau

R = 2.0
δ = 1.0
pml = AnnularPML(R, δ)

k = 1.0
a = two_mode_pole_coef(0:1, 2.0+1.0im)

u = HankelSeries(k, a)

# intrp = interpolation(u, pml, range(-τ/8, τ/8, length=11), 0.999; ε=1e-8);
intrp = interpolation(u, pml, range(-0.15, -0.1, length=5), 0.99; ε=1e-8);

integrand(ν::Number, ∂tν_∂ν::Number) = 1.0/∂tν_∂ν + ((1-ν)^2)*∂tν_∂ν
integrand(p::InterpPoint) = integrand(p.ν, p.∂tν_∂ν)

integrand_patch_fnc(patch, ζ0, ζ1, ν, ζ) = integrand(evalute_and_correct(u, pml, patch, ζ0, ζ1, ν, ζ))

integral_h = integrate_hcubature(intrp.continuous_region[1], integrand_patch_fnc; rtol=1e-2, maxevals=1_000)
