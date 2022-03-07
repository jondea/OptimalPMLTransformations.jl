using Revise
using OptimalPMLTransformations
using Tau
using Plots

R = 2.0
δ = 1.0
pml = AnnularPML(R, δ)

k = 1.0

a = two_mode_pole_coef(0:1, 2.0+1.0im)
u = HankelSeries(k, a)

intrp = interpolate(u, pml, range(-τ/8, τ/8, length=11), 0.999; ε=1e-8);

integrand(ν::Number, ∂tν_∂ν::Number) = 1.0/∂tν_∂ν + ((1-ν)^2)*∂tν_∂ν
integrand(p::InterpPoint) = integrand(p.ν, p.∂tν_∂ν)

int_order = 2

region1 = intrp.continuous_region[1]
line1 = deepcopy(region1.lines[1])

num_points = Int[]
vals = ComplexF64[]

push!(num_points, length(line1.points))
push!(vals, integrate(line1, integrand; order=int_order))

for _ in 1:10
    refine!(line1, u, pml)
    push!(num_points, length(line1.points))
    push!(vals, integrate(line1, integrand; order=int_order))
end

plot(num_points[1:end-1], abs.(vals[1:end-1] .- vals[end]), scale=:log10)
