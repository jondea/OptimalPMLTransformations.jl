### A Pluto.jl notebook ###
# v0.18.0

using Markdown
using InteractiveUtils

# ╔═╡ 0de7404b-46bd-4462-b6f9-413a86cde6a2
begin
    import Pkg
    # activate the shared project environment
    Pkg.activate(Base.current_project())
    # instantiate, i.e. make sure that all packages are downloaded
    Pkg.instantiate()

	using PlutoUI, PlutoLinks
	using LinearAlgebra
	using SpecialFunctions
	using Gridap
	import Gridap: Point
	using Gridap.Fields
	using Gridap.Geometry
    using Tau
    using Colors
	using GridapGmsh
	using StaticArrays
	using WGLMakie,GridapMakie
end

# ╔═╡ 918fcc72-bf9a-43e7-b13a-7ce76738767c
@revise using OptimalPMLTransformations

# ╔═╡ cb382b58-16e1-45ab-8099-6f812c956071
md"""
# PML Helmholtz equation in Cartesian Annulus

"""

# ╔═╡ da50bbc7-403e-490d-a30e-b7a4efb89a8c
md"""
## Physical parameters

We define the geometry and physics parameters.
"""

# ╔═╡ 2279ecc7-6526-4819-a5ea-e40a66f7d95a
k = 2.0

# ╔═╡ 33ea3131-bdb4-45a3-973c-e4c60a4c645b
λ = 2π/k

# ╔═╡ be580a34-c070-481c-ad6f-ffed29e6fc3b
R = 4.0          # Width of the area

# ╔═╡ 3d2e2bff-4d55-42e6-8ff8-540be800dca0
r_c = 1.0          # Radius of the cylinder

# ╔═╡ 034816e3-8e5b-4409-b52b-e06fe8cd2933
δ_pml = 1.0      # Thickness of the PML

# ╔═╡ 85abed5a-5cd2-4b89-8928-07c8f48de368
md"""
## Discrete Model
"""

# ╔═╡ 25f93e89-3c00-4e40-a7a0-93e0d8a8ce9b
resol = 10.0      # Number of points per wavelength in mesh

# ╔═╡ 12e2d719-c74a-4632-85d5-a8c253ba7ade
l_c = λ/resol      # Characteristic length

# ╔═╡ 91c19722-cbc3-46fb-acff-dc674e94bf48
periodic_θ=(false, true)

# ╔═╡ 2ee30f63-0535-4bb5-8bb0-85e6112f394e
rθ_to_xy = (rθ)->Point(rθ[1]*cos(rθ[2]), rθ[1]*sin(rθ[2]))

# ╔═╡ 356a4ad6-4d28-479d-8a59-468bb59529bf
model = CartesianDiscreteModel(CartesianDescriptor(Point(r_c,0.0), Point(R+δ_pml, τ), (20,60); map = rθ_to_xy, isperiodic=periodic_θ))

# ╔═╡ 3a00e5aa-957e-40f6-a444-7f076ca4c4e9
let
	Ωp = Triangulation(model) # No method yet to plot CartesianGrid?
	fig = plot(Ωp)
	wireframe!(Ωp, color=RGBA(0,0,0,0.4), linewidth=0.5)
	cam2d!(fig.figure.scene, reset = Keyboard.left_control)
	# scatter!(Ω, marker=:star8, markersize=20, color=:blue)
	fig
end

# ╔═╡ 49ee6e73-ccf3-47cf-b99c-60f79b5727c2
md"""
## FE spaces

### Test and trial finite element function space
"""

# ╔═╡ 821a015b-e8e1-47a9-9124-d2431ac6ac75
order = 3

# ╔═╡ 371d409a-447a-46da-88bf-bc6e24c48c7d
reffe = ReferenceFE(lagrangian,Float64,order)

# ╔═╡ 9ec6187a-1393-4496-b508-fc54294a32f0
V = TestFESpace(model,reffe;conformity=:H1, dirichlet_tags="boundary", vector_type=Vector{ComplexF64})

# ╔═╡ 042aaaf8-3b64-44e8-b26c-245960ca85fd
md"We apply unit Dirichlet conditions on the inner cylinder, and zero Dirichlet conditions on the outer PML boundary."

# ╔═╡ 13b018b6-3379-4c1b-9dca-a5157e9cd62b
u_D(x) = (1.0 + 0.0im) * (norm(x) < r_c*1.1)

# ╔═╡ 58e98595-6015-44ef-8f20-f44cc954b2fe
U = TrialFESpace(V,u_D)

# ╔═╡ ddead707-c0f6-4e9e-a12c-0b69eae32d74
md"""
## Numerical integration

"""

# ╔═╡ c4b1c3ae-1e5c-43aa-97ac-33eb17cbaa6a
degree = order*2

# ╔═╡ 8051d4fb-620c-4426-a0b7-a6350dd0803c
md"Why do we have to use triangulation and not the elements directly?"

# ╔═╡ 6ce7d52b-b7a3-46d7-8966-86b93228b04d
Ω = Triangulation(model)

# ╔═╡ 5fb2241c-3bf0-4a02-8234-b165d6259c9c
dΩ = Measure(Ω,degree)

# ╔═╡ 5e790bdf-ab97-4fe4-9af5-3670474bb604
md"""
## PML formulation

"""

# ╔═╡ c5ad1c02-9d85-449b-83e8-2bb90a1a2a63
pml = SFB(AnnularPML(R, δ_pml),k)

# ╔═╡ 8cb354e5-448d-47c3-bb67-f5ac5992144d
md"""
## Weak form

Construct the bi-linear weak form of the Helmholtz equation

"""

# ╔═╡ 0ae4b233-3943-4ee4-b593-4f7bdd4dcdfa
md"No source"

# ╔═╡ 542439b4-d337-4d9d-8323-089a8c33fe96
b(v) = 0

# ╔═╡ 81a2859c-71d0-4a59-a7fb-9cd77d50b4d7
md"""
## Solve

We assemble the finite element operator in Gridap with the bi-linear and linear form, then solve for the field.

"""

# ╔═╡ 7cfe7698-c04e-4b6a-bbde-9de54b93df3e
md"""
## Comparison with analytical solution
"""

# ╔═╡ 1f7fd331-e3e3-4e79-a1b0-1544f5ab5678
md"Value of hankel function at inner boundary, which we will use to normalise the field to be 1 at inner boundary."

# ╔═╡ f625145b-5988-407f-8e9c-be47781ac941
h0 = hankelh1(0,k*r_c)

# ╔═╡ 6ce73844-b3f4-425b-b8d2-d23440e4b879
md"Construct the analytical solution in finite element basis"

# ╔═╡ 5a391731-d63e-44f2-a723-6e669d8f4341
uh_t = CellField(x->hankelh1(0,k*norm(x))/h0, Ω)

# ╔═╡ 3d6df51a-91af-427a-8847-9908e0b49980
uh_t.cell_field

# ╔═╡ 7a453df6-9187-4b9d-a223-b49461119f6f
print(uh_t)

# ╔═╡ 2c4db0e3-1e2d-4f85-87e8-e52c243e252a
# ### Compare the difference in the bulk
function Bulk(x)
    if norm(x)<=R
        return 1
    else
        return 0
    end
end

# ╔═╡ 7aa86c69-c366-4a0d-8b8d-36756975b15e
md"## Dependencies"

# ╔═╡ 75193c36-91cc-4582-a5b3-d8992eff2484
import OptimalPMLTransformations: jacobian

# ╔═╡ cb384c0d-6e33-4d68-85e9-924ada655dda
J = x -> begin
	if norm(x) <= R
		return TensorValue{2,2,ComplexF64}(1,0,0,1)
	else
		xs = CartesianCoordinates(x[1], x[2])
		j = jacobian(pml, xs)
		return TensorValue{2,2,ComplexF64}(j[1,1], j[1,2], j[2,1], j[2,2])
	end
end

# ╔═╡ eb27ab8a-ad56-451f-9568-c97690e844ce
# Plain Helmholtz:
# a(u,v) = ∫( ∇(v)⋅∇(u) - k^2*(v*u))*dΩ
a(u,v) = ∫( ( ((inv∘J)⋅∇(v))⋅((inv∘J)⋅∇(u)) - (k^2*(v*u)) )*(det∘J))*dΩ

# ╔═╡ 709ed3be-b0f3-42b5-930b-a136c6803165
op = AffineFEOperator(a,b,U,V)

# ╔═╡ ad22f43c-64b9-4c17-92eb-0d9a8d264f4c
uh = solve(op)

# ╔═╡ 3e36acc8-c468-4c11-a3cb-0fe0f9cece4d
let
	fig = plot(Ω, abs(uh))
	# fig = plot(Ω, real(uh))
	wireframe!(Ω, color=RGBA(0,0,0,0.5), linewidth=0.5)
	cam2d!(fig.figure.scene, reset = Keyboard.left_control)
	# scatter!(Ω, marker=:star8, markersize=20, color=:blue)
	fig
end

# ╔═╡ ee9c4d2b-705b-4ff3-84a2-75cf10938dc1
writevtk(Ω,"annulus_xy",cellfields=["Real"=>real(uh),
        "Imag"=>imag(uh),
        "Norm"=>abs2(uh),
        # "Arg"=>angle(uh),
        ])

# ╔═╡ 89204ee5-b7cb-4828-aed1-c625fab3d4fd
Difference=sqrt(sum(∫(abs2(uh_t-uh)*Bulk)*dΩ)/sum(∫(abs2(uh_t)*Bulk)*dΩ))

# ╔═╡ 75a8817c-1047-4dca-893b-30dbf3bf1b9c
let
	fig = plot(Ω, abs(uh - uh_t)*Bulk, colormap=:heat)
	# Colorbar(fig[1,2], plt)
	# wireframe!(Ω, color=RGBA(1,1,1,0.1), linewidth=0.5)
	cam2d!(fig.figure.scene, reset = Keyboard.left_control)
	# scatter!(Ω, marker=:star8, markersize=20, color=:blue)
	fig
end

# ╔═╡ be59001e-b529-49ff-beeb-d901d1bc11cb
html"""
<style>
  main {
    max-width: 900px;
  }
</style>
"""

# ╔═╡ Cell order:
# ╟─cb382b58-16e1-45ab-8099-6f812c956071
# ╟─da50bbc7-403e-490d-a30e-b7a4efb89a8c
# ╠═2279ecc7-6526-4819-a5ea-e40a66f7d95a
# ╠═33ea3131-bdb4-45a3-973c-e4c60a4c645b
# ╠═be580a34-c070-481c-ad6f-ffed29e6fc3b
# ╠═3d2e2bff-4d55-42e6-8ff8-540be800dca0
# ╠═034816e3-8e5b-4409-b52b-e06fe8cd2933
# ╟─85abed5a-5cd2-4b89-8928-07c8f48de368
# ╠═25f93e89-3c00-4e40-a7a0-93e0d8a8ce9b
# ╠═12e2d719-c74a-4632-85d5-a8c253ba7ade
# ╠═91c19722-cbc3-46fb-acff-dc674e94bf48
# ╠═2ee30f63-0535-4bb5-8bb0-85e6112f394e
# ╠═356a4ad6-4d28-479d-8a59-468bb59529bf
# ╠═3a00e5aa-957e-40f6-a444-7f076ca4c4e9
# ╟─49ee6e73-ccf3-47cf-b99c-60f79b5727c2
# ╠═821a015b-e8e1-47a9-9124-d2431ac6ac75
# ╠═371d409a-447a-46da-88bf-bc6e24c48c7d
# ╠═9ec6187a-1393-4496-b508-fc54294a32f0
# ╟─042aaaf8-3b64-44e8-b26c-245960ca85fd
# ╠═13b018b6-3379-4c1b-9dca-a5157e9cd62b
# ╠═58e98595-6015-44ef-8f20-f44cc954b2fe
# ╟─ddead707-c0f6-4e9e-a12c-0b69eae32d74
# ╠═c4b1c3ae-1e5c-43aa-97ac-33eb17cbaa6a
# ╟─8051d4fb-620c-4426-a0b7-a6350dd0803c
# ╠═6ce7d52b-b7a3-46d7-8966-86b93228b04d
# ╠═5fb2241c-3bf0-4a02-8234-b165d6259c9c
# ╟─5e790bdf-ab97-4fe4-9af5-3670474bb604
# ╠═c5ad1c02-9d85-449b-83e8-2bb90a1a2a63
# ╠═cb384c0d-6e33-4d68-85e9-924ada655dda
# ╟─8cb354e5-448d-47c3-bb67-f5ac5992144d
# ╠═eb27ab8a-ad56-451f-9568-c97690e844ce
# ╟─0ae4b233-3943-4ee4-b593-4f7bdd4dcdfa
# ╠═542439b4-d337-4d9d-8323-089a8c33fe96
# ╟─81a2859c-71d0-4a59-a7fb-9cd77d50b4d7
# ╠═709ed3be-b0f3-42b5-930b-a136c6803165
# ╠═ad22f43c-64b9-4c17-92eb-0d9a8d264f4c
# ╠═3e36acc8-c468-4c11-a3cb-0fe0f9cece4d
# ╠═ee9c4d2b-705b-4ff3-84a2-75cf10938dc1
# ╟─7cfe7698-c04e-4b6a-bbde-9de54b93df3e
# ╟─1f7fd331-e3e3-4e79-a1b0-1544f5ab5678
# ╠═f625145b-5988-407f-8e9c-be47781ac941
# ╟─6ce73844-b3f4-425b-b8d2-d23440e4b879
# ╠═5a391731-d63e-44f2-a723-6e669d8f4341
# ╠═3d6df51a-91af-427a-8847-9908e0b49980
# ╠═7a453df6-9187-4b9d-a223-b49461119f6f
# ╠═2c4db0e3-1e2d-4f85-87e8-e52c243e252a
# ╠═89204ee5-b7cb-4828-aed1-c625fab3d4fd
# ╠═75a8817c-1047-4dca-893b-30dbf3bf1b9c
# ╟─7aa86c69-c366-4a0d-8b8d-36756975b15e
# ╠═0de7404b-46bd-4462-b6f9-413a86cde6a2
# ╠═918fcc72-bf9a-43e7-b13a-7ce76738767c
# ╠═75193c36-91cc-4582-a5b3-d8992eff2484
# ╠═be59001e-b529-49ff-beeb-d901d1bc11cb
