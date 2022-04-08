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
	using Gridap.Fields
	using Gridap.Geometry
    using Tau
    using Colors
	using GridapGmsh
    import GridapGmsh: gmsh,GmshDiscreteModel
	using StaticArrays
	using WGLMakie,GridapMakie
end


# ╔═╡ cb382b58-16e1-45ab-8099-6f812c956071
md"""
# PML Helmholtz equation in 2D

"""

# ╔═╡ da50bbc7-403e-490d-a30e-b7a4efb89a8c
md"""
## Physical parameters

We define the geometry and physics parameters.
"""

# ╔═╡ 2279ecc7-6526-4819-a5ea-e40a66f7d95a
k = 1.0

# ╔═╡ 33ea3131-bdb4-45a3-973c-e4c60a4c645b
λ = 2π/k

# ╔═╡ be580a34-c070-481c-ad6f-ffed29e6fc3b
X = 4.0          # Width of the area

# ╔═╡ b848e3c7-3b46-41e0-aa78-ff4fcfcb0857
Y = 4.0          # Height of the area

# ╔═╡ 3d2e2bff-4d55-42e6-8ff8-540be800dca0
r_c = 1.0          # Radius of the cylinder

# ╔═╡ 034816e3-8e5b-4409-b52b-e06fe8cd2933
δ_pml = 1.0      # Thickness of the PML

# ╔═╡ 85abed5a-5cd2-4b89-8928-07c8f48de368
md"""
## Discrete Model

We import the model from the `geometry.msh` mesh file using the `GmshDiscreteModel` function defined in `GridapGmsh`. The mesh file is created with GMSH in Julia.
"""

# ╔═╡ 4146c37f-e088-473c-b11d-fb926ea4a5c7
function generate_and_save_mesh(X,Y,a,δ_pml,lc)
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.clear()
    gmsh.model.add("geometry")

    function addAndSetPhyicalGroupName(dims, tags, name)
        gmsh.model.setPhysicalName(dims, gmsh.model.addPhysicalGroup(dims, tags), name)
    end

    function addBox(x1, x2, y1, y2, lc, name)
        p11 = gmsh.model.geo.addPoint(x1, y1, 0, lc)
        p21 = gmsh.model.geo.addPoint(x2, y1, 0, lc)
        p22 = gmsh.model.geo.addPoint(x2, y2, 0, lc)
        p12 = gmsh.model.geo.addPoint(x1, y2, 0, lc)

        l1 = gmsh.model.geo.addLine(p11, p21)
        l2 = gmsh.model.geo.addLine(p21, p22)
        l3 = gmsh.model.geo.addLine(p22, p12)
        l4 = gmsh.model.geo.addLine(p12, p11)

        curve_loop_id = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
        addAndSetPhyicalGroupName(1, [l1, l2, l3, l4], name)
        return curve_loop_id
    end

    function addCircle(x_centre, y_centre, radius, lc, name)
        p_l = gmsh.model.geo.addPoint(x_centre-radius, y_centre, 0, lc)
        p_c = gmsh.model.geo.addPoint(x_centre,        y_centre, 0, lc)
        p_r = gmsh.model.geo.addPoint(x_centre+radius, y_centre, 0, lc)

        arc1 = gmsh.model.geo.addCircleArc(p_l, p_c, p_r)
        arc2 = gmsh.model.geo.addCircleArc(p_r, p_c, p_l)

        curve_loop_id = gmsh.model.geo.addCurveLoop([arc1, arc2])
        addAndSetPhyicalGroupName(1, [arc1, arc2], name)
        return curve_loop_id
    end

    cylinder_curve_loop_id = addCircle(0.0, 0.0, a, lc, "Cylinder")
    bulk_curve_loop_id = addBox(-X, X, -Y, Y, lc, "BulkOuter")
    pml_curve_loop_id = addBox(-X-δ_pml, X+δ_pml, -Y-δ_pml, Y+δ_pml, lc, "PMLOuter")

    pml_plane_surface_id = gmsh.model.geo.addPlaneSurface([pml_curve_loop_id,bulk_curve_loop_id])
    bulk_plane_surface_id = gmsh.model.geo.addPlaneSurface([bulk_curve_loop_id,cylinder_curve_loop_id])
    addAndSetPhyicalGroupName(2, [pml_plane_surface_id, bulk_plane_surface_id], "WholeMesh")
    addAndSetPhyicalGroupName(2, [pml_plane_surface_id], "PML")
    addAndSetPhyicalGroupName(2, [bulk_plane_surface_id], "Bulk")

    gmsh.model.geo.synchronize()

    # We can then generate a 2D mesh...
    gmsh.model.mesh.generate(2)
    # ... and save it to disk
    gmsh.write("geometry.msh")
    gmsh.finalize()
end

# ╔═╡ 25f93e89-3c00-4e40-a7a0-93e0d8a8ce9b
resol = 10.0      # Number of points per wavelength in mesh

# ╔═╡ 12e2d719-c74a-4632-85d5-a8c253ba7ade
l_c = λ/resol      # Characteristic length

# ╔═╡ 4e9bd8da-1e16-4ada-88da-676d5562827d
generate_and_save_mesh(X,Y,r_c,δ_pml,l_c)

# ╔═╡ 356a4ad6-4d28-479d-8a59-468bb59529bf
model = GmshDiscreteModel("geometry.msh")

# ╔═╡ 6d0dc467-df67-4404-a7b3-748ed2008325
typeof(model.grid)

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
V = TestFESpace(model,reffe,dirichlet_tags=["Cylinder", "PMLOuter"], vector_type=Vector{ComplexF64})

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

# ╔═╡ 6ce7d52b-b7a3-46d7-8966-86b93228b04d
Ω = Triangulation(model)

# ╔═╡ 3a00e5aa-957e-40f6-a444-7f076ca4c4e9
let
	fig = plot(Ω)
	wireframe!(Ω, color=RGBA(0,0,0,0.4), linewidth=0.5)
	cam2d!(fig.figure.scene, reset = Keyboard.left_control)
	# scatter!(Ω, marker=:star8, markersize=20, color=:blue)
	fig
end

# ╔═╡ 5fb2241c-3bf0-4a02-8234-b165d6259c9c
dΩ = Measure(Ω,degree)

# ╔═╡ 5e790bdf-ab97-4fe4-9af5-3670474bb604
md"""
## PML formulation

"""

# ╔═╡ 5a20ed81-bbb2-4564-a0b6-f2b207a6d3e6
begin
	struct PMLTransformation<:Function
	    k::Float64
		X::SVector{2,Float64}
		δ::Float64
	end
	function x_to_ν(pml::PMLTransformation, x)
		(abs.(x) .- abs.(pml.X))./pml.δ
	end
	function tx(pml::PMLTransformation, ν::Number)
		k = pml.k; δ = pml.δ
		ν > 0 ? X .- im/pml.k*log.(1-ν) : one(im*ν)
	end
	function (pml::PMLTransformation)(x)
		ν = x_to_ν(pml, x)
		tx = X .- im/pml.k*log.(1-ν)
	    return VectorValue(tx[1],tx[2])
	end
	function ∂tx_∂x(pml::PMLTransformation, ν::Number)
		k = pml.k; δ = pml.δ
		ν > 0 ? im/(k*δ)/(1-ν) : one(im*ν)
	end
	function jacobian(pml::PMLTransformation)
		return x -> begin
			ν = x_to_ν(pml, SVector(x[1], x[2]))
			return TensorValue{2,2,ComplexF64}(∂tx_∂x(pml,ν[1]), 0, 0, ∂tx_∂x(pml,ν[2]))
		end
	end
	function Fields.∇(pml::PMLTransformation)
		jacobian(pml)
	end
end

# ╔═╡ 8dc64274-8c24-4c48-b260-50478a3e9c6f
pml = PMLTransformation(k, SVector(X,Y), δ_pml)

# ╔═╡ 8cb354e5-448d-47c3-bb67-f5ac5992144d
md"""
## Weak form

Construct the bi-linear weak form of the Helmholtz equation

"""

# ╔═╡ cb384c0d-6e33-4d68-85e9-924ada655dda
J = jacobian(pml)

# ╔═╡ eb27ab8a-ad56-451f-9568-c97690e844ce
# Plain Helmholtz: a(u,v) = ∫( ∇(v)⋅∇(u) - k^2*(v*u))*dΩ
a(u,v) = ∫( ( ((inv∘J)⋅∇(v))⋅((inv∘J)⋅∇(u)) - (k^2*(v*u)) )*(det∘J))*dΩ

# ╔═╡ 0ae4b233-3943-4ee4-b593-4f7bdd4dcdfa
md"No source"

# ╔═╡ 542439b4-d337-4d9d-8323-089a8c33fe96
b(v) = 0

# ╔═╡ 81a2859c-71d0-4a59-a7fb-9cd77d50b4d7
md"""
## Solve

We assemble the finite element operator in Gridap with the bi-linear and linear form, then solve for the field.

"""

# ╔═╡ 709ed3be-b0f3-42b5-930b-a136c6803165
op = AffineFEOperator(a,b,U,V)

# ╔═╡ ad22f43c-64b9-4c17-92eb-0d9a8d264f4c
uh = solve(op)

# ╔═╡ 3e36acc8-c468-4c11-a3cb-0fe0f9cece4d
let
	fig = plot(Ω, real(uh), aspectratio=1.0)
	wireframe!(Ω, color=RGBA(1,1,1,0.1), linewidth=0.5)
	cam2d!(fig.figure.scene, reset = Keyboard.left_control)
	# scatter!(Ω, marker=:star8, markersize=20, color=:blue)
	fig
end

# ╔═╡ ee9c4d2b-705b-4ff3-84a2-75cf10938dc1
writevtk(Ω,"demo",cellfields=["Real"=>real(uh),
        "Imag"=>imag(uh),
        "Norm"=>abs2(uh),
        # "Arg"=>angle(uh),
        ])

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
    if abs(x[1])<X && abs(x[2])<Y
        return 1
    else
        return 0
    end
end

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

# ╔═╡ 7aa86c69-c366-4a0d-8b8d-36756975b15e
md"## Dependencies"

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
# ╠═b848e3c7-3b46-41e0-aa78-ff4fcfcb0857
# ╠═3d2e2bff-4d55-42e6-8ff8-540be800dca0
# ╠═034816e3-8e5b-4409-b52b-e06fe8cd2933
# ╟─85abed5a-5cd2-4b89-8928-07c8f48de368
# ╠═4146c37f-e088-473c-b11d-fb926ea4a5c7
# ╠═25f93e89-3c00-4e40-a7a0-93e0d8a8ce9b
# ╠═12e2d719-c74a-4632-85d5-a8c253ba7ade
# ╠═4e9bd8da-1e16-4ada-88da-676d5562827d
# ╠═356a4ad6-4d28-479d-8a59-468bb59529bf
# ╠═3a00e5aa-957e-40f6-a444-7f076ca4c4e9
# ╠═6d0dc467-df67-4404-a7b3-748ed2008325
# ╟─49ee6e73-ccf3-47cf-b99c-60f79b5727c2
# ╠═821a015b-e8e1-47a9-9124-d2431ac6ac75
# ╠═371d409a-447a-46da-88bf-bc6e24c48c7d
# ╠═9ec6187a-1393-4496-b508-fc54294a32f0
# ╟─042aaaf8-3b64-44e8-b26c-245960ca85fd
# ╠═13b018b6-3379-4c1b-9dca-a5157e9cd62b
# ╠═58e98595-6015-44ef-8f20-f44cc954b2fe
# ╟─ddead707-c0f6-4e9e-a12c-0b69eae32d74
# ╠═c4b1c3ae-1e5c-43aa-97ac-33eb17cbaa6a
# ╠═6ce7d52b-b7a3-46d7-8966-86b93228b04d
# ╠═5fb2241c-3bf0-4a02-8234-b165d6259c9c
# ╟─5e790bdf-ab97-4fe4-9af5-3670474bb604
# ╠═5a20ed81-bbb2-4564-a0b6-f2b207a6d3e6
# ╠═8dc64274-8c24-4c48-b260-50478a3e9c6f
# ╟─8cb354e5-448d-47c3-bb67-f5ac5992144d
# ╠═cb384c0d-6e33-4d68-85e9-924ada655dda
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
# ╠═be59001e-b529-49ff-beeb-d901d1bc11cb
