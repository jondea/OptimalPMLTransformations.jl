### A Pluto.jl notebook ###
# v0.18.0

using Markdown
using InteractiveUtils

# ╔═╡ 62714408-c270-40a5-be3a-bfe618da1051
begin
	import Pkg
	Pkg.activate(mktempdir())
end

# ╔═╡ 8c51cda7-0ea0-4c00-891d-fc9bd2134c85
begin
	using Revise
	Pkg.add(url="https://github.com/jondea/InverseHankelFunction.jl")
	# Pkg.develop(url="https://github.com/jondea/OptimalPMLTransformations.jl")
	Pkg.develop(path="/home/jondea/phd/OptimalPMLTransformations.jl")
	using OptimalPMLTransformations
end

# ╔═╡ d6be7663-668d-4b50-b5d2-2f28ab97907a
begin
    Pkg.add(["Tau", "CubicHermiteSpline","FastGaussQuadrature"])
    using Tau, CubicHermiteSpline, FastGaussQuadrature
end

# ╔═╡ 1865da27-c3bf-4a86-9a39-59d1cb5eb9d6
begin
    Pkg.add(["IterTools"])
    using IterTools
end

# ╔═╡ 03f266ac-06b7-11ec-0f78-bd1c4b925d06
begin
    Pkg.add(["Plots", "PlotlyJS"])
    using Plots
	plotlyjs()
end

# ╔═╡ 2923b1e2-bdfd-42f9-b955-ca9dc565497e
using LinearAlgebra

# ╔═╡ 2dcb0cb8-0d06-4d55-9bf9-ed27b5c5bc7e
md"# Adaptive interpolation and integration of optimal PML transformation"

# ╔═╡ 99cbcc12-ceae-4d7e-8b20-ed8d60de0a9c
md"""
## Prerequisites

- We can calculate the optimal transformation with numerical continuation and Newton like root finding
- We can find rips, and we know their local behaviour
"""

# ╔═╡ e84ff3cc-818a-4b0d-a882-fc5850ba96be
md"""
## Setup
"""

# ╔═╡ fbba2e4b-6f86-4aef-a8c7-6d84edb6a80f
md"""
We will demstrate the interpolation and integration with an example.
We first define an annular PML
"""

# ╔═╡ b5e7b62d-dc9d-436f-b369-0cea44f96010
pml = let
	R = 2.0
	δ = 1.0
	AnnularPML(R, δ)
end

# ╔═╡ 4ecbe70a-3af6-4328-9d8b-7181cb41b535
md"pick a wavenumber"

# ╔═╡ 80f13c88-aa00-4000-808b-b7ec35b3aa02
k = 1.0

# ╔═╡ 2a717cb2-82e1-4470-b871-9dea2bd70e2c
md"""
and a field, which will consist of two radial Fourier modes chosen so that there is a pole near to our transformation.
Indeed so near that it ends up crossing it and creating a rip.
We do this so that we can properly test out our interpolation and integration schemes on a difficult example. 
"""

# ╔═╡ 0d75f7cb-6503-412d-b4c7-76772d05ff33
u = let
	a = two_mode_pole_coef(0:1, 2.0+1.0im)
	HankelSeries(k, a)
end

# ╔═╡ 3c75cc96-a8d8-4f9d-b78d-3205108e1686
md"## Find rips"

# ╔═╡ 611b863a-796f-4b42-8eab-f4b31888e140
rips = classify_outer_boundary(u, pml, 0, 2π; Nζ=101, ε=1e-5, δ=1e-1, verbose=false)

# ╔═╡ d8c6957a-8cbd-422b-b77c-4f842b7238d2
md"""
## Approximation through the PML

When calculating the transformation at a point within the PML (ν, θ), we take small steps from the inner PML boundary (because there we know that u=U).
As we step along, we try to ensure continuity.
So when the transformation changes rapidly, or the corrector does not converge, we reduce the step size.
The points along the way naturally give us adaptive knot points for our interpolation.
We use the adaptivity of the solver to refine the approximation when the transformation changes rapidly.

We could linearly interpolate between these knot points, but we can do a bit better.
When we evaluate the transformation at a point, we get the derivatives (in ν and θ) for free (relatively speaking).
So we use these at each point to construct a cubic Hermite interpolation.

The plot below shows the points where the transformation has been evaluated, the solid line is the cubic Hermite interpolation, and the dashed line is a piecewise linear interpolation.
"""

# ╔═╡ 429d80ae-7fcf-4240-a6f0-e955ba158b3b
let	
	ν_vec = Float64[]
	tν_vec = ComplexF64[]
	∂tν_∂ν_vec = ComplexF64[]
	
	ν_max = 0.999
	θ = rips[2].ζ+0.001
	optimal_pml_transformation_solve(u, pml, ν_max, θ, ν_vec, tν_vec, ∂tν_∂ν_vec)
	intrp = CubicHermiteSplineInterpolation(ν_vec, tν_vec, ∂tν_∂ν_vec)
	
	ν_vec_plot = 0:0.001:ν_max
	complex2cols(c) = [real.(c) imag.(c)]
	
	colors=[:orange :blue]
	
	plot(legend=false)
	plot!(ν_vec, complex2cols(tν_vec), label=["real(linear approx)" "imag(linear approx)"], color=colors, linestyle=:dash)
	plot!(ν_vec_plot, complex2cols(intrp.(ν_vec_plot)), label=["real(hermite approx)" "imag(hermite approx)"], color=colors)
	scatter!(ν_vec, complex2cols(intrp.(ν_vec)), label=["real(knot)" "imag(knot)"], color=colors)
end

# ╔═╡ df0c1768-feb6-4c9d-9c3b-cd06781e6ab0
md"""
The transformation can be unbounded, specifically at the outer edge.
Furthermore, the derivatives of the transformation can be unbounded at the tip of the rip.
When determining the optimal transformation, if either quanties cannot be determined, we mark them as NaN.
"""

# ╔═╡ 5a99aa34-151f-40b1-95b5-eba87ee4d5fb
md"""
## Approximation through and across the PML

We have a way to approximate the transformation through the PML (ν), how do we build an approximation through *and* across the PML (ν,θ)?




To approximate the transformation across the PML, we start with several approximations through the PML.
We then compare these approximations through the PML, and if the maximum relative difference between them is too large, we find the approximation between the two.
Then we perform the same procedure recursively between both initial approximations and the middle one.
If the difference persists, we stop recursively subdividing when the difference in ζ is less than some previously defined δ.
"""

# ╔═╡ f5faa6af-9bb2-41ec-9517-579fc41db3e2
md"### Plots of transformation"

# ╔═╡ 420cb51f-ee3e-48e7-a34b-6c3932d76970
md"""
### Plots of derivative of transformation
Note the singularity near the tip of the rip
"""

# ╔═╡ 2bb46928-5731-4a50-be2c-c20c5c243fe0
md"""
### Plot of objective of rip tip

Rips happen when ∂u/∂r̃ = 0, so this is what we look for roots of to find the tip of the rip.
"""

# ╔═╡ 7e27197b-5846-4929-aad1-906fcec388be
intrp = interpolate(u, pml, -τ/8:τ/32:τ/8, 0.99)

# ╔═╡ ee98c760-e894-4b6a-9128-7838966bb838
intrp.continuous_region[1].lines[1].tν.x

# ╔═╡ 3528e9f7-75b6-4e7b-9937-de7dbdcf1bc9
intrp.continuous_region[1].lines[1].tν

# ╔═╡ 1e43a2e2-ebc9-4db2-8978-01bee664a02a


# ╔═╡ f69cf5df-49b9-495a-b769-d283aea4e29a
gausslegendre(5)

# ╔═╡ 815b73d9-fa9a-4d4e-97ad-5daf1b8da5d9
md"""
## Technical bits 
"""

# ╔═╡ 6a9a44c6-83e0-4756-941b-7e271756d4f3
function interpolate_and_plot(u::AbstractFieldFunction, pml::PMLGeometry, seriestype::Symbol, νs::AbstractVector, ζs; f=(l,ν)->l(ν), kwargs...)
	intrp = interpolate(u, pml, ζs, maximum(νs))
	plot()
	for region in intrp.continuous_region
		intrp_grid = [f(line, ν) for line in region.lines, ν in νs]
		ζs = [line.ζ for line in region.lines]
		plot!(νs, ζs, abs.(intrp_grid);seriestype)
	end
	plot!()
end

# ╔═╡ 53a2dc51-6684-468d-b95a-6fea0937269e
interpolate_and_plot(u, pml, :heatmap, 0:0.005:0.99, -1.0:0.1:1.0)

# ╔═╡ e57e4592-4e66-45ca-b0ad-7fd32e1bdd5d
interpolate_and_plot(u, pml, :surface, 0:0.005:0.95, -τ/2:0.1:τ/2)

# ╔═╡ dfd329dd-c1a7-4e75-8f74-84a554af4f2c
interpolate_and_plot(u, pml, :surface, 0:0.005:0.95, -τ/8:0.1:τ/8; f=∂tν_∂ν)

# ╔═╡ 468bfd51-7eb3-48a2-b9a0-0c14749d3546
interpolate_and_plot(u, pml, :surface, 0:0.005:0.95, -τ/8:0.1:τ/8; f=(l,ν)->abs(∂u_∂tr(u, PolarCoordinates(pml.R + l(ν), l.ζ))) )

# ╔═╡ 15c527af-a483-4bfa-95e0-374b0dbbd22f
md"## To go in library"

# ╔═╡ b1d5ef17-7a3e-4a8e-8387-a04d75018fa0
#Do not export
function interpolant(PMLFieldFunction, ζ::AbstractVector)
end

# ╔═╡ 404da5d5-737d-4df9-b3e1-6fac3038eb5c
struct PMLFieldFunction{F<:AbstractFieldFunction,P<:PMLGeometry}
	u::F
	pml::P
end

# ╔═╡ 8e497f14-8f77-45b9-8843-bc65e20a61d5
u_pml = PMLFieldFunction(u, pml)

# ╔═╡ 535dc5ec-506a-4afe-9d44-7657deb852f8
#Do not export
function interpolant(PMLFieldFunction, ζ::Number; ν_max=1.0)
	νs = Float64[]
	tνs = ComplexF64[]
	∂tν_∂νs = ComplexF64[]
	∂tν_∂ζs = ComplexF64[]
	optimal_pml_transformation_solve(u_pml.u, u_pml.pml, ν_max, ζ, νs, tνs, ∂tν_∂νs, ∂tν_∂ζs; silent_failure=true)
	return InterpLine(ζ, νs, tνs, ∂tν_∂νs, ∂tν_∂ζs)
end

# ╔═╡ ec66435f-aeb2-472e-91f4-d510d801cdf0
function find_rip(u_pml::PMLFieldFunction, ν0::Real, ζ0::Real, tν0::Number; ε=1e-12)

    ν = ν0
    ζ = ζ0
    tν = tν0

    x = SA[ν, ζ, real(tν), imag(tν)]

	U, dU_dζ = u_pml(NamedTuple{(:u, :∂u_∂tζ)}, 0.0, ζ)
	u, ∂u_∂tν, ∂u_∂tζ, ∂2u_∂tν2, ∂2u_∂tν∂tζ = u_pml(NamedTuple{(:u, :∂u_∂tν, :∂u_∂tζ, :∂2u_∂tν2, :∂2u_∂tν∂tζ)}, tν, ζ)

    # Create vector of residuals
	o_rip = ∂u_∂tν
	o_opt = u - U*(1-ν)
    r = SA[real(o_rip), imag(o_rip), real(o_opt), imag(o_opt)]
	
    n_iter = 1
    while maximum(abs.(r)) > ε*abs(U) && n_iter < 1000

        # Create Jacobian of objectives and unknowns
        ∂o_opt_∂ζ = ∂u_∂tζ - ∂u_∂tζ*(1-ν)
        J = SA[
            0        real(∂2u_∂tν∂tζ)  real(∂2u_∂tν2)  -imag(∂2u_∂tν2);
            0        imag(∂2u_∂tν∂tζ)  imag(∂2u_∂tν2)   real(∂2u_∂tν2);
            real(u)  real(∂o_opt_∂ζ)   real(∂u_∂tν)    -imag(∂u_∂tν)  ;
            imag(u)  imag(∂o_opt_∂ζ)   imag(∂u_∂tν)     real(∂u_∂tν)
        ]

        # Perform Newton step
        x = x - J\r

        # Get values of unknowns from vector
        ν = x[1]
        ζ = x[2]
        tν = x[3] + im*x[4]

        # Recompute field and residual at new point
		U, dU_dζ = u_pml(NamedTuple{(:u, :∂u_∂tζ)}, 0.0, ζ)
		u, ∂u_∂tν, ∂u_∂tζ, ∂2u_∂tν2, ∂2u_∂tν∂tζ= u_pml(NamedTuple{(:u, :∂u_∂tν, :∂u_∂tζ, :∂2u_∂tν2, :∂2u_∂tν∂tζ)}, tν, ζ)

		o_rip = ∂u_∂tν
		o_opt = u - U*(1-ν)
	    r = SA[real(o_rip), imag(o_rip), real(o_opt), imag(o_opt)]
        n_iter += 1
    end

    return ν, ζ, tν
end

# ╔═╡ 02105c89-d058-4cea-ae12-0481eacb6fdc
consecutive_pairs(r) = partition(r, 2, 1)

# ╔═╡ 76701429-c0dd-454e-98fd-4e27a097898b
function Base.argmin(f::Function, l1::InterpLine, l2::InterpLine)
	f1, i1 = findmin(f, l1.points)
	f2, i2 = findmin(f, l2.points)
	f1 < f2 ? l1.points[i1] : l2.points[i2]
end

# ╔═╡ 973df994-bb15-4681-a6e6-bde44766c046
function relative_l2_difference(line1::InterpLine, line2::InterpLine)
	knots = 0:0.01:1.0
	line1_points = line1.(knots)
	line2_points = line2.(knots)
	rel_diff = (2*norm(line1_points - line2_points)
		/(norm(line1_points) + norm(line2_points)))
	return rel_diff
end

# ╔═╡ a74672a6-008a-43bc-b07e-1581064f3d02
function find_rips!(rips::Vector{Rip2D}, u_pml::PMLFieldFunction, ζs::AbstractVector; ε=1e-5, δ=1e-1, tν_metric=relative_l2_difference)::Vector{Rip2D}

	tν₋ = interpolant(u_pml, first(ζs))
	for (ζ₋, ζ₊) in consecutive_pairs(ζs)
		tν₊ = interpolant(u_pml, first(ζs))
		if abs(ζ₊ - ζ₋) < δ
			rip_guess = argmin(p->p.∂u_∂tν, tν₋, tν₊)
			νᵣ, ζᵣ, tνᵣ = find_rip(u_pml, rip_guess.ν, rip_guess.ζ, rip_guess.tν)
			push!(rips, Rip2D(νᵣ, ζᵣ, tνᵣ))
		elseif tν_metric(tν₋,tν₊) > ε
			find_rips!(rips, range(ζ₋, ζ₊, length=3); ε, δ)
		end
		tν₋ = tν₊
	end
	
    return rips

end

# ╔═╡ d68dbba7-eac0-44c2-af0c-08fa5638e954
function find_rips(args...; kwargs...)::Vector{Rip2D}
	rips = Vector{Rip2D}(undef,0)
	find_rips!(rips, args...; kwargs...)
end

# ╔═╡ 92a2f127-bdaa-4ab9-90f3-68c2b6b0fc49
find_rips(u_pml, range(0, τ, length=5))

# ╔═╡ 8f2e628f-14d3-4fe1-941d-d6c99caa7679
rip_objective(field::Union{NamedTuple,InterpPoint}) = field.∂u_∂tν

# ╔═╡ bb96f896-48e0-4d38-8ea2-85378ea4c6cc
optimal_transformation_objective(u, U, ν) = u - U*(1-ν)

# ╔═╡ Cell order:
# ╟─2dcb0cb8-0d06-4d55-9bf9-ed27b5c5bc7e
# ╟─99cbcc12-ceae-4d7e-8b20-ed8d60de0a9c
# ╟─e84ff3cc-818a-4b0d-a882-fc5850ba96be
# ╟─fbba2e4b-6f86-4aef-a8c7-6d84edb6a80f
# ╠═b5e7b62d-dc9d-436f-b369-0cea44f96010
# ╟─4ecbe70a-3af6-4328-9d8b-7181cb41b535
# ╠═80f13c88-aa00-4000-808b-b7ec35b3aa02
# ╟─2a717cb2-82e1-4470-b871-9dea2bd70e2c
# ╠═0d75f7cb-6503-412d-b4c7-76772d05ff33
# ╠═8e497f14-8f77-45b9-8843-bc65e20a61d5
# ╟─3c75cc96-a8d8-4f9d-b78d-3205108e1686
# ╠═611b863a-796f-4b42-8eab-f4b31888e140
# ╠═92a2f127-bdaa-4ab9-90f3-68c2b6b0fc49
# ╠═a74672a6-008a-43bc-b07e-1581064f3d02
# ╠═d68dbba7-eac0-44c2-af0c-08fa5638e954
# ╠═ec66435f-aeb2-472e-91f4-d510d801cdf0
# ╟─d8c6957a-8cbd-422b-b77c-4f842b7238d2
# ╠═429d80ae-7fcf-4240-a6f0-e955ba158b3b
# ╟─df0c1768-feb6-4c9d-9c3b-cd06781e6ab0
# ╠═5a99aa34-151f-40b1-95b5-eba87ee4d5fb
# ╟─f5faa6af-9bb2-41ec-9517-579fc41db3e2
# ╠═53a2dc51-6684-468d-b95a-6fea0937269e
# ╠═e57e4592-4e66-45ca-b0ad-7fd32e1bdd5d
# ╟─420cb51f-ee3e-48e7-a34b-6c3932d76970
# ╠═dfd329dd-c1a7-4e75-8f74-84a554af4f2c
# ╟─2bb46928-5731-4a50-be2c-c20c5c243fe0
# ╠═468bfd51-7eb3-48a2-b9a0-0c14749d3546
# ╠═7e27197b-5846-4929-aad1-906fcec388be
# ╠═ee98c760-e894-4b6a-9128-7838966bb838
# ╠═3528e9f7-75b6-4e7b-9937-de7dbdcf1bc9
# ╠═1e43a2e2-ebc9-4db2-8978-01bee664a02a
# ╠═f69cf5df-49b9-495a-b769-d283aea4e29a
# ╟─815b73d9-fa9a-4d4e-97ad-5daf1b8da5d9
# ╠═62714408-c270-40a5-be3a-bfe618da1051
# ╠═8c51cda7-0ea0-4c00-891d-fc9bd2134c85
# ╠═d6be7663-668d-4b50-b5d2-2f28ab97907a
# ╠═1865da27-c3bf-4a86-9a39-59d1cb5eb9d6
# ╠═03f266ac-06b7-11ec-0f78-bd1c4b925d06
# ╠═2923b1e2-bdfd-42f9-b955-ca9dc565497e
# ╠═6a9a44c6-83e0-4756-941b-7e271756d4f3
# ╟─15c527af-a483-4bfa-95e0-374b0dbbd22f
# ╠═535dc5ec-506a-4afe-9d44-7657deb852f8
# ╠═b1d5ef17-7a3e-4a8e-8387-a04d75018fa0
# ╠═404da5d5-737d-4df9-b3e1-6fac3038eb5c
# ╠═02105c89-d058-4cea-ae12-0481eacb6fdc
# ╠═76701429-c0dd-454e-98fd-4e27a097898b
# ╠═973df994-bb15-4681-a6e6-bde44766c046
# ╠═8f2e628f-14d3-4fe1-941d-d6c99caa7679
# ╠═bb96f896-48e0-4d38-8ea2-85378ea4c6cc
