### A Pluto.jl notebook ###
# v0.19.5

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
    import FerriteViz, GLMakie
end

# ╔═╡ c80b6d31-ba85-4d8a-bc28-20dde3f74985
@revise using Ferrite

# ╔═╡ c194b0aa-1ecc-40e8-9193-e6147198bfc2
k = 2.0

# ╔═╡ 320885af-a850-4aa5-b8ad-371063acf39b
R = 4.0

# ╔═╡ 5504bcf5-924b-4c7d-8fbd-0458b5b5892b
pml_δ = 1.0

# ╔═╡ 5f59a261-e21a-4d1d-857a-429cc220f6c2
δ_pml = pml_δ

# ╔═╡ 6e5f6ba3-6def-42be-8a54-3e54d24d902f
resolution = 10
N_θ = 5*resolution

# ╔═╡ 5810d1fe-0355-466a-a917-4b590042d364
N_r = 3*resolution

# ╔═╡ 8ac284f0-7116-411a-aeb9-09061e74422f
N_pml = 1*resolution

# ╔═╡ af73f242-133c-41cd-883a-4a4a3cba52a0
cylinder_radius = 1.0

# ╔═╡ 6bcd6d30-3a54-4e34-bbdf-a2b177368905
# const ∇ = Tensors.gradient

# ╔═╡ ea410c14-d99b-4d12-a3a1-086529ea213b
reshape([Node((x,y)) for x in 0:0.2:1, y in 0:0.5:3],:)

# ╔═╡ c323e48f-d5ea-4a4f-8827-b3fc6e163cd0
skipfirst(v) = v[begin+1:end]

# ╔═╡ ab4ff9e7-e92c-4319-bb31-adea87fb16e5
flatten(v) = reshape(v,:)

# ╔═╡ 0a6cf79e-d111-4f89-b3be-9e2ffc9b5558
equality_constraint(free_node_id::Int, constrained_node_id::Int) = AffineConstraint(constrained_node_id, [free_node_id=>1.0+0.0im], 0.0im)

# ╔═╡ 1071b515-29e3-4abf-a4c9-33890d37f571
# const Δ = Tensors.hessian;
# Modify this to be generic in order, with a stretching and hard code the periodicity
# QuadraticQuadrilateral
function generate_pml_grid(::Type{QuadraticQuadrilateral}, N_θ, N_r, N_pml, r_inner, R, δ_pml)
    ncell_x = (N_r + N_pml)
 	ncell_y = N_θ
	nodes_x = vcat(range(r_inner,R,length=2*N_r+1), skipfirst(range(R,R+δ_pml,length=2*N_pml+1)))
	nodes_y = range(0,τ,length=2*N_θ+1)

	# Generate nodes
    nodes = [Node((x,y)) for x in nodes_x, y in nodes_y]

	# Generate cells
	node_idxs = reshape(1:length(nodes), size(nodes))
    cells = [
		QuadraticQuadrilateral(
			(node_idxs[2*i-1,2*j-1], node_idxs[2*i+1,2*j-1], node_idxs[2*i+1,2*j+1],
			 node_idxs[2*i-1,2*j+1], node_idxs[2*i  ,2*j-1], node_idxs[2*i+1,2*j],
			 node_idxs[2*i,  2*j+1], node_idxs[2*i-1,2*j],   node_idxs[2*i  ,2*j]))
		for i in 1:ncell_x, j in 1:ncell_y
	]

    # Cell faces
	cell_idxs = reshape(1:length(cells), size(cells))
	bottom_face = [FaceIndex(cl, 1) for cl in cell_idxs[:,    begin]]
	right_face  = [FaceIndex(cl, 2) for cl in cell_idxs[end,  :    ]]
	top_face    = [FaceIndex(cl, 3) for cl in cell_idxs[:,    end  ]]
	left_face   = [FaceIndex(cl, 4) for cl in cell_idxs[begin,:    ]]

	boundary = [bottom_face; right_face; top_face; left_face]
    boundary_matrix = Ferrite.boundaries_to_sparse(boundary)

    # Cell facesets
    facesets = Dict{String, Set{FaceIndex}}(
		"bottom" => Set{FaceIndex}(bottom_face),
		"right"  => Set{FaceIndex}(right_face),
		"top"    => Set{FaceIndex}(top_face),
		"left"   => Set{FaceIndex}(left_face),
	)

    # nodesets
    nodesets = Dict(
        "bottom" => Set(node_idxs[:,    begin]),
        "right"  => Set(node_idxs[end,  :    ]),
        "top"    => Set(node_idxs[:,    end  ]),
        "left"   => Set(node_idxs[begin,:    ]),
    )

    grid = Grid(flatten(cells), flatten(nodes); boundary_matrix)

    # We should be able to use nodesets and facesets for this, but it doesn't seems to work...
    onbottom(x) = x[2]≈0
    ontop(x) = x[2]≈τ
    onleft(x) = x[1]≈r_inner
    onright(x) = x[1]≈(R+δ_pml)

    addfaceset!(grid, "bottom", onbottom)
    addfaceset!(grid, "right", onright)
    addfaceset!(grid, "top", ontop)
    addfaceset!(grid, "left", onleft)

    # addedgeset!(grid, "bottom", onbottom)
    # addedgeset!(grid, "right", onright)
    # addedgeset!(grid, "top", ontop)
    # addedgeset!(grid, "left", onleft)

    addnodeset!(grid, "bottom", onbottom)
    addnodeset!(grid, "right", onright)
    addnodeset!(grid, "top", ontop)
    addnodeset!(grid, "left", onleft)

    periodic_constraints = [equality_constraint(tn, bn) for (tn, bn) in zip(node_idxs[:,end], node_idxs[:,begin])][begin+1:end-1]

    return grid, periodic_constraints
end

# ╔═╡ 0e7cb220-a7c3-41b5-abcc-a2dcc9111ca1
grid, periodic_constraints = generate_pml_grid(QuadraticQuadrilateral, N_θ, N_r, N_pml, cylinder_radius, R, δ_pml)

# ╔═╡ 0aa48490-c2c9-4ed6-b60f-49c76d237b9c
#grid = generate_grid(Quadrilateral, (50, 50), Vec{2}((cylinder_radius,0.0)), Vec{2}((R+pml_δ,τ)))

# ╔═╡ 00be45a6-881a-4bad-bd2c-d28a20bb95bc
grid.nodes

# ╔═╡ 713a5e7f-2200-466e-b5bc-6a18746e4a3e
dim = 2

# ╔═╡ 97120304-9920-42f3-a0b4-0ee6f71457f7
ip = Lagrange{dim, RefCube, 2}()

# ╔═╡ 768d2be4-85e3-4015-9955-fa1682c8fd90
qr = QuadratureRule{dim, RefCube}(4)

# ╔═╡ 2f7b496d-bb59-4631-8533-5754e9541109
qr_face = QuadratureRule{dim-1, RefCube}(4)

# ╔═╡ dd572d4c-d1c5-44d6-9dcb-8742100d29f5
cellvalues = CellScalarValues(qr, ip);

# ╔═╡ 4b68a4cf-3276-42d8-abf2-3dd7a9000899
facevalues = FaceScalarValues(qr_face, ip);

# ╔═╡ fa126b60-a706-4159-bbfc-39bff0a6929d
dh = let
	dh = DofHandler(ComplexF64, grid)
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

function get_nodes_on_face(dh::DofHandler, fi::FaceIndex)
    (cellidx, faceidx) = fi
    _celldofs = celldofs!(dh, cellidx) # extract the dofs for this cell
    r = local_face_dofs_offset[faceidx]:(local_face_dofs_offset[faceidx+1]-1)
    _celldofs[local_face_dofs[r]]
end

# Extend to covr dh with multiple fields
function nodedof(dh::DofHandler, cell_idx::Int, node_idx)
    # @assert isclosed(dh)
    return dh.cell_dofs[dh.cell_dofs_offset[cell_idx] + node_idx - 1]
end

Base.@kwdef mutable struct PeriodicNodeMatch
    # Coordinate along the face for the two nodes
    face_coord::Float64
    # Index of the degree of freedom which we will constrain to be equal to the free dof
    constrained_dof_index::Int = -1
    # Index of the degree of freedom which will set the value of the constrined dof
    free_dof_index::Int = -1
end

function insert_constrained!(v::Vector{PeriodicNodeMatch}, face_coord, constrained_dof_index)
    i = searchsortedfirst(v, PeriodicNodeMatch(;face_coord); by=m->m.face_coord)
    if i <= length(v) && v[i].face_coord ≈ face_coord
        v[i].constrained_dof_index = constrained_dof_index
    elseif i > 1 && v[i-1].face_coord ≈ face_coord
        v[i-1].constrained_dof_index = constrained_dof_index
    else
        insert!(v,i,PeriodicNodeMatch(;face_coord, constrained_dof_index))
    end
end

function insert_free!(v::Vector{PeriodicNodeMatch}, face_coord, free_dof_index)
    i = searchsortedfirst(v, PeriodicNodeMatch(;face_coord); by=m->m.face_coord)
    if i <= length(v) && v[i].face_coord ≈ face_coord
        v[i].free_dof_index = free_dof_index
    elseif i > 1 && v[i-1].face_coord ≈ face_coord
        v[i-1].free_dof_index = free_dof_index
    else
        insert!(v,i,PeriodicNodeMatch(;face_coord, free_dof_index))
    end
end

function node_indices(interpolation, fi::FaceIndex)
    face_idx = fi.idx[2]
    return Ferrite.faces(interpolation)[face_idx]
end

getnode(grid, cell_idx, node_idx) = grid.nodes[grid.cells[cell_idx].nodes[node_idx]]

function add_periodic!(ch, free_to_constrained_facesets, global_to_face_coord, field_name=nothing)

    if isnothing(field_name) && length(ch.dh.field_names) == 1
        field_idx = 1
    else
        field_idx = Ferrite.find_field(ch.dh, field_name)
    end
    interpolation = Ferrite.getfieldinterpolation(ch.dh, field_idx)

    for f_to_c_faceset in free_to_constrained_facesets
        free_faceset, constrained_faceset = f_to_c_faceset

        matches = Vector{PeriodicNodeMatch}()
        for free_face in free_faceset
            cell_idx = free_face.idx[1]
            for node_idx in node_indices(interpolation, free_face)
                coord = getnode(dh.grid, cell_idx, node_idx).x
                free_dof_index = nodedof(dh, cell_idx, node_idx)
                insert_free!(matches, global_to_face_coord(coord), free_dof_index)
            end
        end

        for constrained_face in constrained_faceset
            cell_idx = constrained_face.idx[1]
            for node_idx in node_indices(interpolation, constrained_face)
                coord = getnode(dh.grid, cell_idx, node_idx).x
                constrained_dof_index = nodedof(dh, cell_idx, node_idx)
                insert_constrained!(matches, global_to_face_coord(coord), constrained_dof_index)
            end
        end

        for match in matches
            if match.constrained_dof_index < 1 || match.free_dof_index < 1
                warning("Invalid dof index for ", match)
            end
            add!(ch, AffineConstraint(match.constrained_dof_index, [match.free_dof_index=>1.0], 0.0))
        end
    end
    return ch
end


# ╔═╡ 53ae2bac-59b3-44e6-9d6c-53a18567ea2f
ch = let
	ch = ConstraintHandler(dh)

    Ferrite.getfaceset(grid::Grid, faceset_names::Vector{<:AbstractString}) = union(getfaceset.(Ref(grid), faceset_names)...)

    # pdbc = PeriodicDirichlet(:u, ["bottom" => "top"])
	# add!(ch, pdbc)
    add_periodic!(ch, [getfaceset(grid, ["bottom"])=>getfaceset(grid, ["top"])], x->x[1])

    dbc = Dirichlet(:u, union(getfaceset.(Ref(grid), ["left","right"])...), (x,t) -> u_ana(x))
	add!(ch, dbc)

    close!(ch)
	update!(ch, 0.0)
	ch
end


# ╔═╡ baca19e6-ee39-479e-9bd5-f5838cc9f869
function dof_to_coord_dict(dh::DofHandler)
    global_dofs = zeros(Int, ndofs_per_cell(dh))
    dof_to_node = Dict{Int,Vec{2, Float64}}()
    for cell in CellIterator(dh)
        coords = getcoordinates(cell)
        celldofs!(global_dofs, cell)
        for (global_dof, coord) in zip(global_dofs, coords)
            dof_to_node[global_dof] = coord
        end
    end
    dof_to_node
end

function plot_dof_numbers!(dh::DofHandler)
    for (dof, node) in dof_to_coord_dict(dh)
        GLMakie.text!([string(dof)], position = [(node[1], node[2])])
    end
end

# ╔═╡ baca19e6-ee39-479e-9bd5-f5838cc9f869
function doassemble(cellvalues::CellScalarValues{dim}, facevalues::FaceScalarValues{dim},
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
            f_true = zero(T)
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
	K = create_sparsity_pattern(dh, ch)
	K, f = doassemble(cellvalues, facevalues, K, dh)
    # I think the affine contraint handler might be broken... something to do with global => localdofs
	# Or should AffineConstraints be efined in terms of dof number not node id?
    apply!(K, f, ch)
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

function plot_u(fnc=real)
    plotter = FerriteViz.MakiePlotter(dh,fnc.(u))
    FerriteViz.surface(plotter,field=:u)
end

FerriteViz.wireframe(grid,markersize=5,strokewidth=1,celllabels=true,facelabels=true)
plot_dof_numbers!(dh)

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
# ╠═5f59a261-e21a-4d1d-857a-429cc220f6c2
# ╠═6e5f6ba3-6def-42be-8a54-3e54d24d902f
# ╠═5810d1fe-0355-466a-a917-4b590042d364
# ╠═8ac284f0-7116-411a-aeb9-09061e74422f
# ╠═af73f242-133c-41cd-883a-4a4a3cba52a0
# ╠═6bcd6d30-3a54-4e34-bbdf-a2b177368905
# ╠═ea410c14-d99b-4d12-a3a1-086529ea213b
# ╠═c323e48f-d5ea-4a4f-8827-b3fc6e163cd0
# ╠═ab4ff9e7-e92c-4319-bb31-adea87fb16e5
# ╠═1071b515-29e3-4abf-a4c9-33890d37f571
# ╠═0a6cf79e-d111-4f89-b3be-9e2ffc9b5558
# ╠═0e7cb220-a7c3-41b5-abcc-a2dcc9111ca1
# ╠═0aa48490-c2c9-4ed6-b60f-49c76d237b9c
# ╠═00be45a6-881a-4bad-bd2c-d28a20bb95bc
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
