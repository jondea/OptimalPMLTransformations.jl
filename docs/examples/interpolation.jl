### A Pluto.jl notebook ###
# v0.15.1

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
	Pkg.develop(url="https://github.com/jondea/OptimalPMLTransformations.jl")
	using OptimalPMLTransformations
end

# ╔═╡ d6be7663-668d-4b50-b5d2-2f28ab97907a
begin
    Pkg.add(["Tau", "CubicHermiteSpline"])
    using Tau, CubicHermiteSpline
end

# ╔═╡ 03f266ac-06b7-11ec-0f78-bd1c4b925d06
begin
    Pkg.add(["Plots", "PlotlyJS"])
    using Plots
	plotlyjs()
end

# ╔═╡ 2923b1e2-bdfd-42f9-b955-ca9dc565497e
using LinearAlgebra

# ╔═╡ 422728fc-e266-4594-9a9e-9dc6d9b35058
md"# Adaptive approximation of optimal transformation"

# ╔═╡ e84ff3cc-818a-4b0d-a882-fc5850ba96be
md"""
## Setup
"""

# ╔═╡ fbba2e4b-6f86-4aef-a8c7-6d84edb6a80f
md"""
First define our PML
"""

# ╔═╡ b5e7b62d-dc9d-436f-b369-0cea44f96010
pml = let
	R = 2.0
	δ = 1.0
	AnnularPML(R, δ)
end

# ╔═╡ 4ecbe70a-3af6-4328-9d8b-7181cb41b535
md"wavenumber"

# ╔═╡ 80f13c88-aa00-4000-808b-b7ec35b3aa02
k = 1.0

# ╔═╡ 2a717cb2-82e1-4470-b871-9dea2bd70e2c
md"""
and our field, which will consist of two radial Fourier modes chosen so that there is a pole near to our transformation.
Indeed so near that it will probably cross it and create a rip
"""

# ╔═╡ 0d75f7cb-6503-412d-b4c7-76772d05ff33
u = let
	a = two_mode_pole_coef(0:1, 2.0+1.0im)
	HankelSeries(k, a)
end

# ╔═╡ d8c6957a-8cbd-422b-b77c-4f842b7238d2
md"""
## Approximation through the PML


When we evaluate the transformation at a point, we get the derivatives "for free". So we use these at each point to construct a cubic Hermite interpolation through the PML.
We use the adaptivity of the solver to refine the approximation when the transformation changes rapidly.

The plot below shows the points where the transformation has been evaluated, the solid line is the cubic Hermite interpolation, and the dashed line is a piecewise linear interpolation.
"""

# ╔═╡ 429d80ae-7fcf-4240-a6f0-e955ba158b3b
let	
	ν_vec = Float64[]
	tν_vec = ComplexF64[]
	∂tν_∂ν_vec = ComplexF64[]
	
	ν_max = 0.999
	rips = classify_outer_boundary(u, pml, 0, 2π; Nζ=101, ε=1e-5, δ=1e-1, verbose=true)
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

# ╔═╡ 5a99aa34-151f-40b1-95b5-eba87ee4d5fb
md"""
## Approximation through and across the PML

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

# ╔═╡ Cell order:
# ╟─422728fc-e266-4594-9a9e-9dc6d9b35058
# ╟─e84ff3cc-818a-4b0d-a882-fc5850ba96be
# ╟─fbba2e4b-6f86-4aef-a8c7-6d84edb6a80f
# ╠═b5e7b62d-dc9d-436f-b369-0cea44f96010
# ╟─4ecbe70a-3af6-4328-9d8b-7181cb41b535
# ╠═80f13c88-aa00-4000-808b-b7ec35b3aa02
# ╟─2a717cb2-82e1-4470-b871-9dea2bd70e2c
# ╠═0d75f7cb-6503-412d-b4c7-76772d05ff33
# ╟─d8c6957a-8cbd-422b-b77c-4f842b7238d2
# ╠═429d80ae-7fcf-4240-a6f0-e955ba158b3b
# ╟─5a99aa34-151f-40b1-95b5-eba87ee4d5fb
# ╟─f5faa6af-9bb2-41ec-9517-579fc41db3e2
# ╠═53a2dc51-6684-468d-b95a-6fea0937269e
# ╠═e57e4592-4e66-45ca-b0ad-7fd32e1bdd5d
# ╟─420cb51f-ee3e-48e7-a34b-6c3932d76970
# ╠═dfd329dd-c1a7-4e75-8f74-84a554af4f2c
# ╟─2bb46928-5731-4a50-be2c-c20c5c243fe0
# ╠═468bfd51-7eb3-48a2-b9a0-0c14749d3546
# ╟─815b73d9-fa9a-4d4e-97ad-5daf1b8da5d9
# ╠═62714408-c270-40a5-be3a-bfe618da1051
# ╠═8c51cda7-0ea0-4c00-891d-fc9bd2134c85
# ╠═d6be7663-668d-4b50-b5d2-2f28ab97907a
# ╠═03f266ac-06b7-11ec-0f78-bd1c4b925d06
# ╠═2923b1e2-bdfd-42f9-b955-ca9dc565497e
# ╠═6a9a44c6-83e0-4756-941b-7e271756d4f3
