### A Pluto.jl notebook ###
# v0.18.0

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
    using Tensors
    using Tau
    using SparseArrays
	using SpecialFunctions
    using LinearAlgebra
end

# ╔═╡ c80b6d31-ba85-4d8a-bc28-20dde3f74985
@revise using Ferrite

# ╔═╡ c194b0aa-1ecc-40e8-9193-e6147198bfc2
k = 2.0

# ╔═╡ 320885af-a850-4aa5-b8ad-371063acf39b
R = 4.0

# ╔═╡ 5504bcf5-924b-4c7d-8fbd-0458b5b5892b
pml_δ = 1.0

# ╔═╡ af73f242-133c-41cd-883a-4a4a3cba52a0
cylinder_radius = 1.0

# ╔═╡ 6bcd6d30-3a54-4e34-bbdf-a2b177368905
# const ∇ = Tensors.gradient

# ╔═╡ 1071b515-29e3-4abf-a4c9-33890d37f571
# const Δ = Tensors.hessian;

# ╔═╡ 0aa48490-c2c9-4ed6-b60f-49c76d237b9c
grid = generate_grid(Quadrilateral, (50, 50), Vec{2}((cylinder_radius,0.0)), Vec{2}((R+pml_δ,τ)))

# ╔═╡ 713a5e7f-2200-466e-b5bc-6a18746e4a3e
dim = 2

# ╔═╡ 97120304-9920-42f3-a0b4-0ee6f71457f7
ip = Lagrange{dim, RefCube, 1}()

# ╔═╡ 768d2be4-85e3-4015-9955-fa1682c8fd90
qr = QuadratureRule{dim, RefCube}(2)

# ╔═╡ 2f7b496d-bb59-4631-8533-5754e9541109
qr_face = QuadratureRule{dim-1, RefCube}(2)

# ╔═╡ dd572d4c-d1c5-44d6-9dcb-8742100d29f5
cellvalues = CellScalarValues(qr, ip);

# ╔═╡ 4b68a4cf-3276-42d8-abf2-3dd7a9000899
facevalues = FaceScalarValues(qr_face, ip);

# ╔═╡ fa126b60-a706-4159-bbfc-39bff0a6929d
dh = let
	dh = DofHandler(grid)
	push!(dh, :u, 1)
	close!(dh)
	dh
end

# ╔═╡ 401cd067-650b-4a86-bf69-6dc202128b47
typeof(dh)

# ╔═╡ c3b9a58a-f191-4a0d-8e1d-dc595de23421
n_h = 3

# ╔═╡ f25ba6a8-cd2f-4f19-8203-979229345386
function u_ana(x::Vec{2, T}) where {T}
    r, θ = x[1], x[2]
    return hankelh1(n_h, k*r) * exp(im*n_h*θ)
end

# ╔═╡ 291b0883-2a04-4dee-bd75-e7553d42f0ec
u_ana(n::Node) = u_ana(n.x)

# ╔═╡ 53ae2bac-59b3-44e6-9d6c-53a18567ea2f
dbcs = let
	dbcs = ConstraintHandler(dh;bctype=ComplexF64)
	dbc = Dirichlet(:u, union(getfaceset(grid, "left"), getfaceset(grid, "bottom"), getfaceset(grid, "top"), getfaceset(grid, "right")), (x,t) -> u_ana(x))
	add!(dbcs, dbc)
	close!(dbcs)
	update!(dbcs, 0.0)
	dbcs
end

# ╔═╡ baca19e6-ee39-479e-9bd5-f5838cc9f869
function doassemble(cellvalues::CellScalarValues{dim}, facevalues::FaceScalarValues{dim},
                         K::SparseMatrixCSC, dh::DofHandler) where {dim}
    b = 1.0
	fill!(K.nzval, zero(ComplexF64))
    f = zeros(ComplexF64, ndofs(dh))
    assembler = start_assemble(K, f)

    n_basefuncs = getnbasefunctions(cellvalues)
    global_dofs = zeros(Int, ndofs_per_cell(dh))

    fe = zeros(ComplexF64, n_basefuncs) # Local force vector
    Ke = zeros(ComplexF64, n_basefuncs, n_basefuncs) # Local stiffness mastrix

    @inbounds for (cellcount, cell) in enumerate(CellIterator(dh))
        fill!(Ke, 0)
        fill!(fe, 0)
        coords = getcoordinates(cell)

        reinit!(cellvalues, cell)
		for q_point in 1:getnquadpoints(cellvalues)
            dΩ = getdetJdV(cellvalues, q_point)
            coords_qp = spatial_coordinate(cellvalues, q_point, coords)
            r = coords_qp[1]
            θ = coords_qp[2]
            Jᵣₓ = diagm(Tensor{2,2}, [1.0, 1/r])
            f_true = zero(ComplexF64)
            for i in 1:n_basefuncs
                δu = shape_value(cellvalues, q_point, i)
                ∇δu = shape_gradient(cellvalues, q_point, i)
                fe[i] += (δu * f_true) * r * dΩ
                for j in 1:n_basefuncs
                    u = shape_value(cellvalues, q_point, j)
                    ∇u = shape_gradient(cellvalues, q_point, j)
                    Ke[i, j] += ((Jᵣₓ ⋅ ∇δu) ⋅ (Jᵣₓ⋅∇u) - k^2*δu * u) * r * dΩ
                end
            end
        end

        celldofs!(global_dofs, cell)
        assemble!(assembler, global_dofs, fe, Ke)
    end
    return K, f
end;

# ╔═╡ 05f429a6-ac35-4a90-a45e-8803c7f1692a
u = let
	# We should be able to remove this at some point
	K = create_sparsity_pattern(dh; field_type=ComplexF64)
	K, f = doassemble(cellvalues, facevalues, K, dh)
	apply!(K, f, dbcs)
	u = K \ f
	u
end

# ╔═╡ 592b59d6-9fbf-4b6a-a6df-553ed40082b4
let
	vtkfile = vtk_grid("helmholtz", dh)
	vtk_point_data(vtkfile, dh, real.(u), "_real")
	vtk_point_data(vtkfile, dh, imag.(u), "_imag")
	# Can't seem to get these to output the cells in the right order...
    u_ana_nodes = u_ana.(dh.grid.nodes)
    vtk_point_data(vtkfile, dh, first.(getcoordinates.(dh.grid.nodes)), "_r")
    vtk_point_data(vtkfile, dh, real.(u_ana_nodes), "_exact_real")
	vtk_point_data(vtkfile, dh, imag.(u_ana_nodes), "_exact_imag")
	vtk_save(vtkfile)
end

# ╔═╡ b5b526dd-37ef-484c-97fd-2305f0d1d714
html"""
<style>
  main {
    max-width: 900px;
  }
</style>
"""

# ╔═╡ Cell order:
# ╠═a51bf9fd-11a6-48ba-9b0d-ab4397be015c
# ╠═c80b6d31-ba85-4d8a-bc28-20dde3f74985
# ╠═c194b0aa-1ecc-40e8-9193-e6147198bfc2
# ╠═320885af-a850-4aa5-b8ad-371063acf39b
# ╠═5504bcf5-924b-4c7d-8fbd-0458b5b5892b
# ╠═af73f242-133c-41cd-883a-4a4a3cba52a0
# ╠═6bcd6d30-3a54-4e34-bbdf-a2b177368905
# ╠═1071b515-29e3-4abf-a4c9-33890d37f571
# ╠═0aa48490-c2c9-4ed6-b60f-49c76d237b9c
# ╠═713a5e7f-2200-466e-b5bc-6a18746e4a3e
# ╠═97120304-9920-42f3-a0b4-0ee6f71457f7
# ╠═768d2be4-85e3-4015-9955-fa1682c8fd90
# ╠═2f7b496d-bb59-4631-8533-5754e9541109
# ╠═dd572d4c-d1c5-44d6-9dcb-8742100d29f5
# ╠═4b68a4cf-3276-42d8-abf2-3dd7a9000899
# ╠═fa126b60-a706-4159-bbfc-39bff0a6929d
# ╠═401cd067-650b-4a86-bf69-6dc202128b47
# ╠═c3b9a58a-f191-4a0d-8e1d-dc595de23421
# ╠═f25ba6a8-cd2f-4f19-8203-979229345386
# ╠═291b0883-2a04-4dee-bd75-e7553d42f0ec
# ╠═53ae2bac-59b3-44e6-9d6c-53a18567ea2f
# ╠═baca19e6-ee39-479e-9bd5-f5838cc9f869
# ╠═05f429a6-ac35-4a90-a45e-8803c7f1692a
# ╠═592b59d6-9fbf-4b6a-a6df-553ed40082b4
# ╠═b5b526dd-37ef-484c-97fd-2305f0d1d714
