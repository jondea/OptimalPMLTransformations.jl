### A Pluto.jl notebook ###
# v0.19.14

using Markdown
using InteractiveUtils

# ╔═╡ c14dcd6a-8d90-4b5c-a12b-39c7c92f2cca
begin
	import Pkg
	Pkg.activate(mktempdir())
end

# ╔═╡ bdf79010-61e1-11ec-2aaf-67cd1ad9b6db
begin
	Pkg.add(["Revise"])
	using Revise
	Pkg.add(url="https://github.com/jondea/InverseHankelFunction.jl")
	# Pkg.develop(url="https://github.com/jondea/OptimalPMLTransformations.jl")
	Pkg.develop(path="/home/jondea/phd/OptimalPMLTransformations.jl")
	using OptimalPMLTransformations
end

# ╔═╡ 91d6b55f-b656-4d0d-8636-df98fb680813
begin
    Pkg.add(["Tau", "CubicHermiteSpline","FastGaussQuadrature"])
    using Tau, CubicHermiteSpline, FastGaussQuadrature
end

# ╔═╡ 7dd4091c-a7bd-4ad6-825b-15817ca682de
begin
    Pkg.add(["PlutoUI"])
    using PlutoUI
end

# ╔═╡ 9f9d612d-4025-4aa6-ae4f-242906b8a36d
begin
    Pkg.add(["Plots", "PlotlyJS"])
    using Plots
	plotlyjs()
end

# ╔═╡ 2fbcb7fa-bece-46f6-ba24-3c1644561b63
using Plots: plot

# ╔═╡ 06c6d138-25f6-4ef3-98ab-e294189449aa
R = 2.0

# ╔═╡ fe7ae947-b180-435d-81c2-c361970d9c67
δ = 1.0

# ╔═╡ 056395c0-dd53-4eba-bbb2-11f55f9d48f1
pml = AnnularPML(R, δ)

# ╔═╡ 727e5c20-76a8-4bd1-9738-07a3d3b592eb
k = 1.0

# ╔═╡ 5be48193-7eb9-4a5f-81ff-61a2a209f09e
a = two_mode_pole_coef(0:1, 2.0+1.0im)

# ╔═╡ 5559db14-bbb0-4ee1-a5a9-d9549fec2d09
u = HankelSeries(k, a)

# ╔═╡ cf33a317-42ac-40cf-93a1-108710869ed7
u_fnc_νζ(tν, ζ) = u(NamedTuple{(:u, :∂u_∂tν, :∂u_∂tζ, :∂2u_∂tν2, :∂2u_∂tν∂tζ, :∂3u_∂tν3)},
	PMLCoordinates(tν,ζ), pml)

# ╔═╡ b8ce3309-982c-4bb3-acff-cab1cddcb447
intrp = interpolation(u, pml, range(-τ/8, τ/8, length=11), 0.999; ε=1e-8)

# ╔═╡ 78bbf375-d069-4181-8115-a10815dba3aa
intrp2 = interpolation(u, pml, range(τ/8, 2τ/8, length=3), 1.0; ε=1e-8)

# ╔═╡ 78dded32-70ea-4b06-a520-02865f710c8a
intrp2.continuous_region

# ╔═╡ 89a90c0c-dba6-43c6-b9c8-8a29cfc58ac1
integrand(ν::Number, ∂tν_∂ν::Number) = 1.0/∂tν_∂ν + ((1-ν)^2)*∂tν_∂ν

# ╔═╡ 2e71c5d6-9406-4eb2-9099-87cb4ceee682
integrand(p::InterpPoint) = integrand(p.ν, p.∂tν_∂ν)

# ╔═╡ 22113d8e-613e-41e9-bb3a-14f13c32f98e
int_order = 2

# ╔═╡ 2fb74755-5938-45b7-ad5c-c06c49e11ba2
region1 = intrp.continuous_region[1]

# ╔═╡ 1a1b34a4-068e-43cc-8067-f3c01a9fc592
md"""
TODO
- Count the number of evaluations of hankel
- Profile different integration schemes, any obvious bits that are slow?
- Where were all the NaNs coming from, and is it an issue?
- Whats happening at infinity?
- Can we solve for the integrand and avoid infinity altogether?
- Compare performance with the old transformed integration method
- Move to a vectorised Hankel function method? CRBond
"""

# ╔═╡ 037c12df-5bb3-40c2-892e-09afcdb27569
import OptimalPMLTransformations.evalute_and_correct

# ╔═╡ 66f5b6d2-cf1a-4e3d-91a9-806d06d0a393
integrand_patch_fnc(patch, ζ0, ζ1, ν, ζ) = integrand(evalute_and_correct(u, pml, patch, ζ0, ζ1, ν, ζ))

# ╔═╡ 93a9e662-2e2b-4a2f-8f00-b864b807b46f
integrate_hcubature(intrp.continuous_region[1], integrand_patch_fnc; atol=1e-14, rtol=1e-12, maxevals=1000_000)

# ╔═╡ d914c19b-6c68-4925-9164-eb06c4909a83
integral_h = integrate_hcubature(intrp, integrand_patch_fnc; atol=1e-14, rtol=1e-12, maxevals=1000_000)

# ╔═╡ f712162a-04b1-4cf9-acfa-ab471111b41b


# ╔═╡ 64277ef1-caf8-40d7-9970-a3f365ab5752
rips = find_rips(u_fnc_νζ, -0.2, 0.0, Nζ=21, ν=0.999, ε=1e-3)

# ╔═╡ b4bbc402-012d-4717-bc0e-2874754c5943
function integrate_trans_gauss(u, pml, ν_range, ζ_range, integration_order)
	field_fnc_νζ(tν, ζ) = u(NamedTuple{(:u, :∂u_∂tν, :∂u_∂tζ, :∂2u_∂tν2, :∂2u_∂tν∂tζ, :∂3u_∂tν3)},
		PMLCoordinates(tν,ζ), pml)

	rips = find_rips(field_fnc_νζ, ζ_range[1], ζ_range[2], Nζ=21, ν=0.999, ε=1e-3)

	ν_crit = only(rips).ν
	ζ_crit = only(rips).ζ

	ζ_width = ζ_range[2]-ζ_range[1]

	# Find singularity
	s_crit = ((ζ_crit - ζ_range[1])/ζ_width, ν_crit)

	# Get Gauss-Legendre knot points and weights, and transform to [0,1]
	nodes, weights = gausslegendre(integration_order)
	nodes .= (nodes .+ 1)/2
	weights .= weights./2

	integral = 0.0 + 0.0im

	n_knot = 1
	for _ in 1:2integration_order # Loop around ζ
		# Initialise stepping through and integrating
		trans_node, _ = gausslegendretrans_mid(n_knot, nodes, weights, s_crit)
		ζ = ζ_range[1] + trans_node[1] * ζ_width
		ν = 0.0
		ν_prev = ν
		tν = 0.0 + 0.0im
		tν_prev = tν
		field_fnc_ν(tν) = field_fnc_νζ(tν, ζ)
		U_field = field_fnc_ν(0.0+0.0im)
		field = U_field
		for _ in 1:2integration_order # Loop around ν
			trans_node, trans_weight = gausslegendretrans_mid(n_knot, nodes, weights, s_crit)
			ν = trans_node[2]
			tν, ∂tν_∂ν, ∂tν_∂ζ, ν_prev, field = optimal_pml_transformation_solve(field_fnc_ν, ν; 					ν0=ν_prev, tν0=tν_prev, field0=field, U_field=U_field, householder_order=3)
			integral += trans_weight*integrand(ν, ∂tν_∂ν)*ζ_width

			n_knot += 1
			ν_prev = ν
			tν_prev = tν
		end
	end
	return integral
end

# ╔═╡ d65778b9-5cfd-4a58-992a-89aeb2a17cbe
integral_g = integrate_trans_gauss(u, pml, (0.0, 1.0), (-τ/8, τ/8), 1000)

# ╔═╡ 057e0a5f-36c2-4ffa-aac3-c377a2866ac9
function integrate_trans_gauss_old(n)

	# Get Gauss-Legendre knot points and weights, and transform to [0,1]
	nodes, weights = gausslegendre(n)
	nodes .= (nodes .+ 1)/2
	weights .= weights./2

	# Initialise stepping through and integrating
	integral = 0.0 + 0.0im

	function int_gauss_trans_2d_mid(f::Function, N::Int; a=0.5, x_crit=(0.7,0.2))
	    nodes, weights = gausslegendreunittrans(N,a)
	    int_f = 0.0
	    for n in 1:4(N^2)
	        node, weight = gausslegendretrans_mid(n, nodes, weights, x_crit)
	        int_f += weight * f(node[1],node[2])
	    end
	    return int_f
	end

	function initiailise(ζ, field_fnc_νζ)
		ν = 0.0
		tν = 0.0 + 0.0im
		field_fnc_ν = field_fnc_νζ(0.0+0.0im)
		U_field = field_fnc_ν(0.0+0.0im)
		return 0.0, 0.0 + 0.0im, field_fnc_ν, U_field
	end

	function step_and_integrand!()
		tν_prev, ∂tν_∂ν, ∂tν_∂ζ, ν_prev, field_prev = exact_mapping_solve_adaptive_steps(field_fnc, ν; 					ν0=ν_prev, tν0=tν_prev, field0=field_prev, U_field=U_field, householder_order=3)
		integral += weight*integrand(ν, ∂tν_∂ν)
	end

	# Step through PML, adding contributions to integral
	for (ν, weight) in zip(nodes, weights)
		initiailise(ζ, field_fnc_νζ)

		for (ν, weight) in zip(nodes, weights)
		end
		for (ν, weight) in zip(nodes, weights)
		end

	end
	return integral
end

# ╔═╡ 46a0e484-2921-4e85-8990-dbe2e19fa6a1


# ╔═╡ b337b5fd-31e8-4b15-89c1-6ec37f7d4ef9
region1.lines[end]

# ╔═╡ 0f01cc3c-50bf-4cbc-9d5a-b11c12042adf
let
	patch = InterpPatch(InterpPoint(0.3508955859812662, -0.47703501586636193 + 0.48277889368037835im, -46.07062793160125 + 236.7295526770068im, -655.3730948368166 - 275.43746576226675im), InterpPoint(0.3508955859812662, -0.47845951656682717 + 0.4823051055739416im, -4.235970269635079 + 518.1163039220834im, -1509.9660764174746 - 300.42793752269847im), InterpPoint(0.35089699857745116, -0.4770804988587099 + 0.48310863407264im, -19.354118901857518 + 229.59130015950115im, -634.7187560106638 - 211.92903162711352im), InterpPoint(0.35089699857745116, -0.47837708868603784 + 0.4829179143945096im, 87.45957830503136 + 363.8450276938105im, -1110.2344129970952 + 54.20936710073559im))
	(ζ0, ζ1, ν0, ν1, I, E) = (-0.1303436279296875, -0.13034210205078128, 0.3508955859812662, 0.35089699857745116, NaN + NaN*im, NaN)

	# f(ν,ζ) = imag(OptimalPMLTransformations.evaluate(patch,ζ0,ζ1,ν,ζ).tν)
	f(ν,ζ) = imag(OptimalPMLTransformations.evalute_and_correct(u, pml, patch, ζ0, ζ1, ν, ζ).tν)

	surface(range(ν0,ν1,length=100), range(ζ0,ζ1,length=100), f)
	(ζ0 - ζ1,  ν0 - ν1)
end

# ╔═╡ f191f9ea-2fc1-4154-852c-778b3785c7a5
md"# Appendix"

# ╔═╡ 802833be-8568-4d8e-b8c0-6fd61f1a8d29
function Plots.plot!(region::OptimalPMLTransformations.ContinuousInterpolation, f::Function, νs::AbstractVector; seriestype=:heatmap, kwargs...)
	intrp_grid = [f.(Ref(line), ν) for line in region.lines, ν in νs]
	ζs = [line.ζ for line in region.lines]
	plot!(νs, ζs, intrp_grid; seriestype)
end

# ╔═╡ b5e2c362-93ec-40b3-b93b-8177119e4b7f
function Plots.plot(intrp::OptimalPMLTransformations.Interpolation, f::Function, νs::AbstractVector; seriestype=:heatmap, kwargs...)
	plot()
	for region in intrp.continuous_region
		plot!(region, f, νs; seriestype, kwargs...)
	end
	plot!()
end

# ╔═╡ e3a153c2-2487-4abc-b2d7-d8584fd9d595
∂tν_∂ν(region1.lines[1], 0.1)

# ╔═╡ 534aeb09-2e3f-42bc-9d56-84539fd12b72
region1.lines[1](0.1)

# ╔═╡ a3d3b5d0-da39-4f37-808e-59577b30fe74
flatten(v) = reshape(v,:)

# ╔═╡ 5dc0c6bd-b40a-4a74-97e5-a6a57ca1c4de
skipfirst(v) = v[begin+1:end]

# ╔═╡ f55e9335-d457-4387-a10c-fe6d5a574487
skiplast(v) = v[begin:end-1]

# ╔═╡ ac123b67-aa73-4f72-a0f7-25e28ce3a2ac
seqdiff(v) = skipfirst(v) .- skiplast(v)

# ╔═╡ efa46f30-4e6e-47b4-9ed8-d0666d03331a
"Return points between each entry of v, scaled by r. Essentially samples of a linear interpolation."
points_between(v, r) = flatten(skiplast(v)' .+ r.*(skipfirst(v) - skiplast(v))')

# ╔═╡ ddcd4435-9612-4605-80d1-170de56b412a
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

# ╔═╡ f1581e91-93cb-4e39-ac69-e9a873b4e33d
plot(region1)

# ╔═╡ 4de66add-1ab6-43c8-9ca5-a48435a6701e
let
	line1 = deepcopy(region1.lines[1])

	num_points = Int[]
	vals = ComplexF64[]

	push!(num_points, length(line1.points))
	push!(vals, integrate(line1, integrand; order=int_order))

	for _ in 1:8
	    refine!(line1, u, pml)
	    push!(num_points, length(line1.points))
	    push!(vals, integrate(line1, integrand; order=int_order))
	end

	plot(num_points[1:end-1], abs.(vals[1:end-1] .- vals[end]), scale=:log10)
end

# ╔═╡ 2cc68d52-cfc9-49fb-ad8a-a7a6affd2617
let
	num_points = Int[]
	vals = ComplexF64[]

	region_refined_in_ζ = deepcopy(intrp.continuous_region[1])
	for i in 0:2
		if i!= 0
			refine_in_ζ!(region_refined_in_ζ, u, pml)
		end
		region = deepcopy(region_refined_in_ζ)
		for line in region.lines
	    	for _ in 1:i refine!(line, u, pml) end
		end
		total_points = mapreduce(l->length(l.points), +, region.lines)
	    push!(num_points, total_points)
	    push!(vals, integrate(region, integrand; order=int_order))
	end

	plot(num_points[1:end-1], abs.(vals[1:end-1] .- vals[end]), scale=:log10), num_points, vals
end

# ╔═╡ e561b9d7-d0ca-456e-951b-f2ec53d9c242
plot(intrp, (line,ν) -> real(∂tν_∂ν(line,ν)), 0:0.01:0.9; seriestype=:surface)

# ╔═╡ 8187b92b-80dc-4dd4-9b82-421550472e87
plot(region1.lines[end])

# ╔═╡ a9b53fb8-5759-4ce2-a0e0-31d259cbb00d
plot(region1.lines[end], p->p.∂tν_∂ν)

# ╔═╡ d84c8962-4f17-4d64-82da-f438ad74c678
let
	νs = [p.ν for p in region1.lines[end].points]
	plot(seqdiff(νs), yscale=:log10, marker=true)
end

# ╔═╡ 9334a0ca-0863-40cf-9a97-f75c499e5858
html"""
<style>
  main {
    max-width: 900px;
  }
</style>
"""

# ╔═╡ 230b3592-c2ac-4e06-9bc9-00ccf75ce3c0
PlutoUI.TableOfContents()

# ╔═╡ dc61296c-5123-4f30-a129-50fc0f0f1b45
html"<button onclick='present()'>present</button>"

# ╔═╡ Cell order:
# ╠═06c6d138-25f6-4ef3-98ab-e294189449aa
# ╠═fe7ae947-b180-435d-81c2-c361970d9c67
# ╠═056395c0-dd53-4eba-bbb2-11f55f9d48f1
# ╠═727e5c20-76a8-4bd1-9738-07a3d3b592eb
# ╠═5be48193-7eb9-4a5f-81ff-61a2a209f09e
# ╠═5559db14-bbb0-4ee1-a5a9-d9549fec2d09
# ╠═cf33a317-42ac-40cf-93a1-108710869ed7
# ╠═b8ce3309-982c-4bb3-acff-cab1cddcb447
# ╠═78bbf375-d069-4181-8115-a10815dba3aa
# ╠═78dded32-70ea-4b06-a520-02865f710c8a
# ╠═89a90c0c-dba6-43c6-b9c8-8a29cfc58ac1
# ╠═2e71c5d6-9406-4eb2-9099-87cb4ceee682
# ╠═22113d8e-613e-41e9-bb3a-14f13c32f98e
# ╠═2fb74755-5938-45b7-ad5c-c06c49e11ba2
# ╠═f1581e91-93cb-4e39-ac69-e9a873b4e33d
# ╠═1a1b34a4-068e-43cc-8067-f3c01a9fc592
# ╠═4de66add-1ab6-43c8-9ca5-a48435a6701e
# ╠═2cc68d52-cfc9-49fb-ad8a-a7a6affd2617
# ╠═037c12df-5bb3-40c2-892e-09afcdb27569
# ╠═66f5b6d2-cf1a-4e3d-91a9-806d06d0a393
# ╠═93a9e662-2e2b-4a2f-8f00-b864b807b46f
# ╠═d914c19b-6c68-4925-9164-eb06c4909a83
# ╠═f712162a-04b1-4cf9-acfa-ab471111b41b
# ╠═64277ef1-caf8-40d7-9970-a3f365ab5752
# ╠═d65778b9-5cfd-4a58-992a-89aeb2a17cbe
# ╠═b4bbc402-012d-4717-bc0e-2874754c5943
# ╠═057e0a5f-36c2-4ffa-aac3-c377a2866ac9
# ╠═e561b9d7-d0ca-456e-951b-f2ec53d9c242
# ╠═46a0e484-2921-4e85-8990-dbe2e19fa6a1
# ╠═8187b92b-80dc-4dd4-9b82-421550472e87
# ╠═a9b53fb8-5759-4ce2-a0e0-31d259cbb00d
# ╠═b337b5fd-31e8-4b15-89c1-6ec37f7d4ef9
# ╠═d84c8962-4f17-4d64-82da-f438ad74c678
# ╠═0f01cc3c-50bf-4cbc-9d5a-b11c12042adf
# ╟─f191f9ea-2fc1-4154-852c-778b3785c7a5
# ╠═2fbcb7fa-bece-46f6-ba24-3c1644561b63
# ╠═b5e2c362-93ec-40b3-b93b-8177119e4b7f
# ╠═802833be-8568-4d8e-b8c0-6fd61f1a8d29
# ╠═e3a153c2-2487-4abc-b2d7-d8584fd9d595
# ╠═534aeb09-2e3f-42bc-9d56-84539fd12b72
# ╠═ddcd4435-9612-4605-80d1-170de56b412a
# ╠═bdf79010-61e1-11ec-2aaf-67cd1ad9b6db
# ╠═c14dcd6a-8d90-4b5c-a12b-39c7c92f2cca
# ╠═91d6b55f-b656-4d0d-8636-df98fb680813
# ╠═7dd4091c-a7bd-4ad6-825b-15817ca682de
# ╠═9f9d612d-4025-4aa6-ae4f-242906b8a36d
# ╠═a3d3b5d0-da39-4f37-808e-59577b30fe74
# ╠═5dc0c6bd-b40a-4a74-97e5-a6a57ca1c4de
# ╠═f55e9335-d457-4387-a10c-fe6d5a574487
# ╠═ac123b67-aa73-4f72-a0f7-25e28ce3a2ac
# ╠═efa46f30-4e6e-47b4-9ed8-d0666d03331a
# ╠═9334a0ca-0863-40cf-9a97-f75c499e5858
# ╠═230b3592-c2ac-4e06-9bc9-00ccf75ce3c0
# ╠═dc61296c-5123-4f30-a129-50fc0f0f1b45
