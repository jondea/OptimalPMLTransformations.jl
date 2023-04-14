### A Pluto.jl notebook ###
# v0.19.14

using Markdown
using InteractiveUtils

# ╔═╡ eb47d80e-9520-11ed-3e4b-11b9e18a2046
begin
    import Pkg
    # activate the shared project environment
    Pkg.activate(@__DIR__)
    # instantiate, i.e. make sure that all packages are downloaded
    Pkg.instantiate()
	using PlutoUI
    using Tau
	using Plots
	import PlotlyJS
	plotlyjs()
    using LinearAlgebra
	using OptimalPMLTransformations
end

# ╔═╡ c6c443b1-0757-44e4-bca3-cb8d68d826dd
k = 10.0

# ╔═╡ 31b42b61-a84a-46b1-98a8-8306923ba32e
n_h = 0

# ╔═╡ 54e42f70-b5ca-47e0-8825-9d735b1cf753
R = 2.0

# ╔═╡ 2b7e5f4b-7d04-4311-8153-cf6260873766
δ_pml=1.0

# ╔═╡ addd5cd7-ba79-4df8-829e-8fefd19cf71e
# u_ana=HankelSeries(k, OffsetVector([1.0], n_h:n_h))
u_ana=two_mode_pole_series(k, R + (1.0+1.0im))

# ╔═╡ 3f492b54-65ef-47e8-b43a-d3069b6b7f27
pml_geom = AnnularPML(R, δ_pml)

# ╔═╡ 9d973e18-92d7-4705-88b5-3caf8176fa2f
u_pml = PMLFieldFunction(u_ana, pml_geom)

# ╔═╡ ba901e21-470e-4156-a184-fa588065b42d
(θ_min, θ_max) = (-0.02, 0.02)

# ╔═╡ 3e97394e-afa3-4dc6-b9c6-60384d912008
intrp = interpolation(u_pml, range(θ_min, θ_max, length=101))

# ╔═╡ b083800b-a3a7-434e-aeef-954ed5d25975
intrp.rips[1]

# ╔═╡ 5ee28cd0-2ef0-4763-83f5-aaf100d0a206
intrp.continuous_region[1]

# ╔═╡ 2c3fbbc1-2b5a-4a0a-8ff7-4283dc402f0d
interpolation(u_pml, range(θ_min, θ_max, length=101))

# ╔═╡ 6e49f5c1-dabc-4ef5-bd4c-8366c01e008a
function Plots.plot!(region::OptimalPMLTransformations.ContinuousInterpolation, f::Function, νs::AbstractVector; seriestype=:heatmap, kwargs...)
	intrp_grid = [f.(Ref(line), ν) for line in region.lines, ν in νs]
	ζs = [line.ζ for line in region.lines]
	plot!(νs, ζs, intrp_grid; seriestype)
end

# ╔═╡ df4df856-73e4-494a-a733-294e8e390e5b
function Plots.plot(line::OptimalPMLTransformations.InterpLine, f::Function=p->p.tν; kwargs...)
	νs = [point.ν for point in line.points]
	y_knots = [f(point) for point in line.points]
	ν_fine = points_between(νs, 0:0.2:1.0)
	y_interps = [f(InterpPoint(line, ν)) for ν in ν_fine]
	if eltype(y_knots) <: Complex
		scatter(νs, [real.(y_knots) imag.(y_knots)], color=[:blue :orange])
		plot!(ν_fine, [real.(y_interps) imag.(y_interps)], color=[:blue :orange])
	else
		plot()
	end
end

# ╔═╡ 1599f867-b81b-4a2d-a7ce-6695d07e1bba
function Plots.plot(intrp::OptimalPMLTransformations.Interpolation, f::Function, νs::AbstractVector; seriestype=:heatmap, kwargs...)
	plot()
	for region in intrp.continuous_region
		plot!(region, f, νs; seriestype, kwargs...)
	end
	plot!()
end

# ╔═╡ fb6bd5d9-b47d-44ce-a671-f17fa39833c2
plot(intrp, (line,ν)->real(line(ν)), 0:0.01:0.99; seriestype=:surface)

# ╔═╡ Cell order:
# ╠═eb47d80e-9520-11ed-3e4b-11b9e18a2046
# ╠═c6c443b1-0757-44e4-bca3-cb8d68d826dd
# ╠═31b42b61-a84a-46b1-98a8-8306923ba32e
# ╠═54e42f70-b5ca-47e0-8825-9d735b1cf753
# ╠═2b7e5f4b-7d04-4311-8153-cf6260873766
# ╠═addd5cd7-ba79-4df8-829e-8fefd19cf71e
# ╠═3f492b54-65ef-47e8-b43a-d3069b6b7f27
# ╠═9d973e18-92d7-4705-88b5-3caf8176fa2f
# ╠═ba901e21-470e-4156-a184-fa588065b42d
# ╠═b083800b-a3a7-434e-aeef-954ed5d25975
# ╠═3e97394e-afa3-4dc6-b9c6-60384d912008
# ╠═5ee28cd0-2ef0-4763-83f5-aaf100d0a206
# ╠═2c3fbbc1-2b5a-4a0a-8ff7-4283dc402f0d
# ╠═fb6bd5d9-b47d-44ce-a671-f17fa39833c2
# ╠═df4df856-73e4-494a-a733-294e8e390e5b
# ╠═1599f867-b81b-4a2d-a7ce-6695d07e1bba
# ╠═6e49f5c1-dabc-4ef5-bd4c-8366c01e008a
