### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ a51bf9fd-11a6-48ba-9b0d-ab4397be015c
begin
    import Pkg
    # activate the shared project environment
    Pkg.activate(Base.current_project())
    # instantiate, i.e. make sure that all packages are downloaded
    Pkg.instantiate()
	using PlutoLinks
	using PlutoUI
    using Tensors
    using Tau
    using SparseArrays
	using SpecialFunctions
    using LinearAlgebra
	# using OptimalPMLTransformations
	using StaticArrays
    import FerriteViz, GLMakie
end

# ╔═╡ c80b6d31-ba85-4d8a-bc28-20dde3f74985
@revise using Ferrite

# ╔═╡ b3e4776d-8d03-47c2-8057-8b6e4f443b90
@revise using OptimalPMLTransformations

# ╔═╡ 424b9b91-9926-4c54-9f9d-329c974a8336
@revise using PMLFerriteExtensions

# ╔═╡ 4c152d80-c523-4da1-9035-7d7b4cb144d9
md"# PML Helmholtz equation in annulus in polar coordinates"

# ╔═╡ 1d9ac613-63c0-4c34-8c84-1af02a5f4172
md"## Parameters"

# ╔═╡ c194b0aa-1ecc-40e8-9193-e6147198bfc2
k = 2.0

# ╔═╡ 320885af-a850-4aa5-b8ad-371063acf39b
R = 4.0

# ╔═╡ 5f59a261-e21a-4d1d-857a-429cc220f6c2
δ_pml = 1.0

# ╔═╡ af5bb610-1c5d-479e-a7d2-318d134ccdd2
resolution = 20

# ╔═╡ 5810d1fe-0355-466a-a917-4b590042d364
N_r = 3*resolution

# ╔═╡ 8ac284f0-7116-411a-aeb9-09061e74422f
N_pml = 3*resolution

# ╔═╡ af73f242-133c-41cd-883a-4a4a3cba52a0
cylinder_radius = 1.0

# ╔═╡ 713a5e7f-2200-466e-b5bc-6a18746e4a3e
dim = 2

# ╔═╡ c3b9a58a-f191-4a0d-8e1d-dc595de23421
n_h = 1

# ╔═╡ fc8b07d5-d035-45c3-ac0b-e7ea29f42bf0
N_θ = 2*max(n_h, 1)*resolution

# ╔═╡ f25ba6a8-cd2f-4f19-8203-979229345386
function u_ana(x::Vec{2, T}) where {T}
    r, θ = x[1], x[2]
    return hankelh1(n_h, k*r) * exp(im*n_h*θ)
end

# ╔═╡ 6ba2449f-ad24-44d0-a551-38f7c1d88f6b
u_ana(x::Vec{2}, _t::Number) = u_ana(x)

# ╔═╡ 70efa29b-880c-4ac5-b9ef-4f3a84af9ab2
u_ana(n::Node) = u_ana(n.x)

# ╔═╡ 314553e9-5c69-4da3-97b0-6287baf20e82
pml = SFB(AnnularPML(R, δ_pml), k)
# pml = InvHankelPML(;R, δ=δ_pml, k, m=n_h)

# ╔═╡ 82e06944-aab0-4486-adc5-7b6b9c7d1275
function J_pml(r, θ)
	rθ = PolarCoordinates(r, θ)
	tr, j = tr_and_jacobian(pml, rθ)
	return tr, Tensors.Tensor{2,2,ComplexF64}(j)
end

# ╔═╡ 608866e5-c10e-4ac1-a120-303869e95c31
in_pml(x::Vec{2}) = x[1] >= R

# ╔═╡ 37eb008f-65fb-42b1-8bd2-14827c6e1e68
in_pml(n::Node) = in_pml(n.x)

# ╔═╡ 0e7cb220-a7c3-41b5-abcc-a2dcc9111ca1
grid = generate_pml_grid(QuadraticQuadrilateral, N_θ, N_r, N_pml, cylinder_radius, R, δ_pml)

# ╔═╡ 97120304-9920-42f3-a0b4-0ee6f71457f7
ip = Lagrange{dim, RefCube, 2}()

# ╔═╡ 768d2be4-85e3-4015-9955-fa1682c8fd90
qr = QuadratureRule{dim, RefCube}(4)

# ╔═╡ d688875d-12f6-491a-89a2-1c346dc26acf
# pml_qr = QuadratureRule{dim, RefCube}(8)
pml_qr = anisotropic_quadrature(RefCube, 4, 4)

# ╔═╡ 2f7b496d-bb59-4631-8533-5754e9541109
qr_face = QuadratureRule{dim-1, RefCube}(4)

# ╔═╡ dd572d4c-d1c5-44d6-9dcb-8742100d29f5
cellvalues = CellScalarValues(qr, ip);

# ╔═╡ e680ed8b-e054-47d2-87b4-ab25c15063a3
pml_cellvalues = CellScalarValues(pml_qr, ip);

# ╔═╡ 4b68a4cf-3276-42d8-abf2-3dd7a9000899
facevalues = FaceScalarValues(qr_face, ip);

# ╔═╡ 0d050dc4-0b97-4a0c-b399-970e1f7284c9
md"## Constraints"

# ╔═╡ 53ae2bac-59b3-44e6-9d6c-53a18567ea2f
function setup_constraint_handler(dh::Ferrite.AbstractDofHandler, left_bc, right_bc)
	ch = ConstraintHandler(dh)

	gridfaceset(s) = getfaceset(grid, s)

    add_periodic!(ch, [gridfaceset("bottom")=>gridfaceset("top")], x->x[1])

    inner_dbc = Dirichlet(:u, gridfaceset("left"), left_bc)
	add!(ch, inner_dbc)

	outer_dbc = Dirichlet(:u, gridfaceset("right"), right_bc)
	add!(ch, outer_dbc)

    close!(ch)
	update!(ch, 0.0)
	ch
end

# ╔═╡ 30826935-619b-460a-afc2-872950052f7a
md"## Assembly"

# ╔═╡ baca19e6-ee39-479e-9bd5-f5838cc9f869
function doassemble(cellvalues::CellScalarValues{dim}, pml_cellvalues::CellScalarValues{dim},
                         K::SparseMatrixCSC, dh::DofHandler) where {dim}
    b = 1.0
    T = dof_type(dh)
	fill!(K.nzval, zero(T))
    f = zeros(T, ndofs(dh))
    assembler = start_assemble(K, f)

    n_basefuncs = getnbasefunctions(cellvalues)
    global_dofs = zeros(Int, ndofs_per_cell(dh))

    fe = zeros(T, n_basefuncs) # Local force vector
    Ke = zeros(T, n_basefuncs, n_basefuncs) # Local stiffness mastrix

    for (cellcount, cell) in enumerate(CellIterator(dh))
        fill!(Ke, 0)
        fill!(fe, 0)
        coords = getcoordinates(cell)
		if in_pml(mean(coords))
			reinit!(pml_cellvalues, cell)
			for q_point in 1:getnquadpoints(pml_cellvalues)
	            dΩ = getdetJdV(pml_cellvalues, q_point)
	            coords_qp = spatial_coordinate(pml_cellvalues, q_point, coords)
	            r = coords_qp[1]
	            θ = coords_qp[2]
				tr, J_pml_ = J_pml(r,θ)
	            Jᵣₓ = diagm(Tensor{2,2}, [1.0, 1/tr])
	            Jₜᵣᵣ = inv(J_pml_)
				detJₜᵣᵣ	= det(J_pml_)
				f_true = zero(T)
	            for i in 1:n_basefuncs
	                δu = shape_value(pml_cellvalues, q_point, i)
	                ∇δu = shape_gradient(pml_cellvalues, q_point, i)
	                fe[i] += (δu * f_true) * tr * dΩ
	                for j in 1:n_basefuncs
	                    u = shape_value(pml_cellvalues, q_point, j)
	                    ∇u = shape_gradient(pml_cellvalues, q_point, j)
	                    Ke[i, j] += ((Jₜᵣᵣ ⋅ (Jᵣₓ ⋅ ∇δu)) ⋅ (Jₜᵣᵣ ⋅ (Jᵣₓ⋅∇u)) - k^2*δu * u
										) * tr * detJₜᵣᵣ * dΩ
	                end
	            end
	        end
		else
			# Bulk
			reinit!(cellvalues, cell)
			for q_point in 1:getnquadpoints(cellvalues)
	            dΩ = getdetJdV(cellvalues, q_point)
				coords_qp = spatial_coordinate(cellvalues, q_point, coords)
	            r = coords_qp[1]
	            for i in 1:n_basefuncs
	                δu = shape_value(cellvalues, q_point, i)
	                ∇δu = shape_gradient(cellvalues, q_point, i)
	                for j in 1:n_basefuncs
	                    u = shape_value(cellvalues, q_point, j)
	                    ∇u = shape_gradient(cellvalues, q_point, j)
	                    Ke[i, j] += (∇δu⋅∇u - k^2*δu * u) * r * dΩ
	                end
	            end
	        end
		end

        celldofs!(global_dofs, cell)
        assemble!(assembler, global_dofs, fe, Ke)
    end
    return K, f
end

# ╔═╡ 8296d8a7-41ce-4111-9564-05e7cdc4bfe8
md"## Solve and plot"

# ╔═╡ 7b7a469a-01a4-4bd4-8d8a-d35f4db70a3c
dh = DofHandler(ComplexF64, grid, [:u])

# ╔═╡ 28991803-729f-4265-a987-fb3a800e26be
ch = setup_constraint_handler(dh, u_ana, (x,t)->zero(ComplexF64))

# ╔═╡ d87c0552-eec5-4765-ad04-03fbab2727fe
function solve(ch, cellvalues, pml_cellvalues)
	K = create_sparsity_pattern(ch.dh, ch)
	K, f = doassemble(cellvalues, pml_cellvalues, K, ch.dh)
    apply!(K, f, ch)
	u = K \ f
	apply!(u, ch)
	return u
end

# ╔═╡ 05f429a6-ac35-4a90-a45e-8803c7f1692a
u = solve(ch, cellvalues, pml_cellvalues )

# ╔═╡ 20c98042-509d-49f0-b011-a45bba3a7dae
write_vtk("helmholtz", dh, u, u_ana)

# ╔═╡ a2c59a93-ed50-44ea-9e7b-8cb99b1f3afb
function plot_u(fnc=real)
    plotter = FerriteViz.MakiePlotter(dh,fnc.(u))
    FerriteViz.surface(plotter,field=:u)
end

# ╔═╡ b9801fdf-6d31-48a1-9837-219199e0f326
md"## Error measure"

# ╔═╡ 0e84a686-6e8f-456c-88c7-1af049e4da96
abs_sq_error = integrate_solution((u,x)-> in_pml(x) ? 0.0 : abs(u - u_ana(x))^2, u, cellvalues, dh)

# ╔═╡ 65b019eb-d01b-4810-991c-59ce48cfe416
abs_sq_norm = integrate_solution((u,x)-> in_pml(x) ? 0.0 : abs(u)^2, u, cellvalues, dh)

# ╔═╡ 339eba6b-be42-4691-a402-ae7ca53e17c6
rel_error = sqrt(abs_sq_error/abs_sq_norm)

# ╔═╡ 27d3071a-722e-4c0d-98b9-cd04d7c7980a
md"# Appendix"

# ╔═╡ 9cfd1f97-f333-40a7-98db-cb9373a857a0
md"## Utils"

# ╔═╡ b5b526dd-37ef-484c-97fd-2305f0d1d714
html"""
<style>
  main {
    max-width: 900px;
  }
</style>
"""

# ╔═╡ 059eceb1-39fd-4a5e-a787-23d2bbf9b547
PlutoUI.TableOfContents()

# ╔═╡ 45be7b0b-c64d-4cfe-a399-de62e6b9094e
md"## Dependencies"

# ╔═╡ Cell order:
# ╟─4c152d80-c523-4da1-9035-7d7b4cb144d9
# ╟─1d9ac613-63c0-4c34-8c84-1af02a5f4172
# ╠═c194b0aa-1ecc-40e8-9193-e6147198bfc2
# ╠═320885af-a850-4aa5-b8ad-371063acf39b
# ╠═5f59a261-e21a-4d1d-857a-429cc220f6c2
# ╠═af5bb610-1c5d-479e-a7d2-318d134ccdd2
# ╠═fc8b07d5-d035-45c3-ac0b-e7ea29f42bf0
# ╠═5810d1fe-0355-466a-a917-4b590042d364
# ╠═8ac284f0-7116-411a-aeb9-09061e74422f
# ╠═af73f242-133c-41cd-883a-4a4a3cba52a0
# ╠═713a5e7f-2200-466e-b5bc-6a18746e4a3e
# ╠═c3b9a58a-f191-4a0d-8e1d-dc595de23421
# ╠═f25ba6a8-cd2f-4f19-8203-979229345386
# ╠═6ba2449f-ad24-44d0-a551-38f7c1d88f6b
# ╠═70efa29b-880c-4ac5-b9ef-4f3a84af9ab2
# ╠═314553e9-5c69-4da3-97b0-6287baf20e82
# ╠═82e06944-aab0-4486-adc5-7b6b9c7d1275
# ╠═608866e5-c10e-4ac1-a120-303869e95c31
# ╠═37eb008f-65fb-42b1-8bd2-14827c6e1e68
# ╠═0e7cb220-a7c3-41b5-abcc-a2dcc9111ca1
# ╠═97120304-9920-42f3-a0b4-0ee6f71457f7
# ╠═768d2be4-85e3-4015-9955-fa1682c8fd90
# ╠═d688875d-12f6-491a-89a2-1c346dc26acf
# ╠═2f7b496d-bb59-4631-8533-5754e9541109
# ╠═dd572d4c-d1c5-44d6-9dcb-8742100d29f5
# ╠═e680ed8b-e054-47d2-87b4-ab25c15063a3
# ╠═4b68a4cf-3276-42d8-abf2-3dd7a9000899
# ╟─0d050dc4-0b97-4a0c-b399-970e1f7284c9
# ╠═53ae2bac-59b3-44e6-9d6c-53a18567ea2f
# ╟─30826935-619b-460a-afc2-872950052f7a
# ╠═baca19e6-ee39-479e-9bd5-f5838cc9f869
# ╟─8296d8a7-41ce-4111-9564-05e7cdc4bfe8
# ╠═7b7a469a-01a4-4bd4-8d8a-d35f4db70a3c
# ╠═28991803-729f-4265-a987-fb3a800e26be
# ╠═05f429a6-ac35-4a90-a45e-8803c7f1692a
# ╠═d87c0552-eec5-4765-ad04-03fbab2727fe
# ╠═20c98042-509d-49f0-b011-a45bba3a7dae
# ╠═a2c59a93-ed50-44ea-9e7b-8cb99b1f3afb
# ╟─b9801fdf-6d31-48a1-9837-219199e0f326
# ╠═0e84a686-6e8f-456c-88c7-1af049e4da96
# ╠═65b019eb-d01b-4810-991c-59ce48cfe416
# ╠═339eba6b-be42-4691-a402-ae7ca53e17c6
# ╟─27d3071a-722e-4c0d-98b9-cd04d7c7980a
# ╟─9cfd1f97-f333-40a7-98db-cb9373a857a0
# ╠═b5b526dd-37ef-484c-97fd-2305f0d1d714
# ╠═059eceb1-39fd-4a5e-a787-23d2bbf9b547
# ╟─45be7b0b-c64d-4cfe-a399-de62e6b9094e
# ╠═a51bf9fd-11a6-48ba-9b0d-ab4397be015c
# ╠═c80b6d31-ba85-4d8a-bc28-20dde3f74985
# ╠═b3e4776d-8d03-47c2-8057-8b6e4f443b90
# ╟─424b9b91-9926-4c54-9f9d-329c974a8336
