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

intrp = interpolation(u, pml, range(-τ/8, τ/8, length=11), 0.999; ε=1e-8);
# intrp = interpolation(u, pml, range(0.5, 0.55, length=2), 0.999; ε=1e-8);

integrand(ν::Number, ∂tν_∂ν::Number) = 1.0/∂tν_∂ν + ((1-ν)^2)*∂tν_∂ν
integrand(p::InterpPoint) = integrand(p.ν, p.∂tν_∂ν)
# ∂integrand_∂ν = -2(1/∂tν_∂ν^2)*∂t2ν_∂ν2 +

int_order = 2

region1 = intrp.continuous_region[1]

integrate(region1, integrand; order=int_order)

refine_in_ζ!(region1, u, pml)

line1 = deepcopy(region1.lines[1])

# tνs = [p.tν for p in line1.points]
# Δtνs = (tνs[2:end] - tνs[1:end-1])./(νs[2:end] - νs[1:end-1])

# num_points = Int[]
# vals = ComplexF64[]

# push!(num_points, length(line1.points))
# push!(vals, integrate(line1, integrand; order=int_order))

# for _ in 1:10
#     refine!(line1, u, pml)
#     push!(num_points, length(line1.points))
#     push!(vals, integrate(line1, integrand; order=int_order))
# end

# plot(num_points[1:end-1], abs.(vals[1:end-1] .- vals[end]), scale=:log10)

# function int_by_refine(n)
#     num_points = Int[]
#     vals = ComplexF64[]

#     region = deepcopy(region1)
#     for i in 0:n
#         if i != 0 refine_in_ζ!(region, u, pml) end
#         push!(num_points, length(region.lines))
#         push!(vals, integrate(region, integrand; order=int_order))
#     end
#     return num_points, vals
# end

# plot(num_points[1:end-1], abs.(vals[1:end-1] .- vals[end]), scale=:log10)

function int_by_refine(n)
	num_points = Int[]
	vals = ComplexF64[]

	for i in 0:n
		region = deepcopy(intrp.continuous_region[1])
		for _ in 0:i refine_in_ζ!(region, u, pml) end
		for line in region.lines
	    	for _ in 0:i refine!(line, u, pml) end
		end
	    push!(num_points, length(region.lines))
	    push!(vals, integrate(region, integrand; order=int_order))
	end

	plot(num_points[1:end-1], abs.(vals[1:end-1] .- vals[end]), scale=:log10)
end


# for _ in 1:3 refine_in_ζ!(region1, u, pml) end
# for line in region1.lines
#     for _ in 1:3 refine!(line, u, pml) end
# end

# integral_r = integrate(region1, integrand; order=int_order)

# import OptimalPMLTransformations.evalute_and_correct
# integrand_patch_fnc(patch, ζ0, ζ1, ν, ζ) = integrand(evalute_and_correct(u, pml, patch, ζ0, ζ1, ν, ζ))

# integral_h = integrate_hcubature(intrp.continuous_region[1], integrand_patch_fnc)

# @show integral_h, integral_r, abs(integral_h - integral_r)/abs(integral_r)
