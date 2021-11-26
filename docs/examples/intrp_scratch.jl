
using OptimalPMLTransformations
using Tau

R = 2.0
δ = 1.0
pml = AnnularPML(R, δ)

k = 1.0

a = two_mode_pole_coef(0:1, 2.0+1.0im)
u = HankelSeries(k, a)

intrp = interpolate(u, pml, range(-τ/8, τ/8, length=11), 0.999; ε=1e-8);

integrand(ν::Number, ∂tν_∂ν::Number) = 1.0/∂tν_∂ν + ((1-ν)^2)*∂tν_∂ν
integrand(p::InterpPoint) = integrand(p.ν, p.∂tν_∂ν)

integrate(intrp.continuous_region[1], integrand; order=2)

region1 = intrp.continuous_region[1]
line1 = region1.lines[1]

integrate(line1, integrand)
