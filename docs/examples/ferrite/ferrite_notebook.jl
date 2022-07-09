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

# ╔═╡ 4c152d80-c523-4da1-9035-7d7b4cb144d9
md"# PML Helmholtz equation in annulus in polar coordinates"

# ╔═╡ 1d9ac613-63c0-4c34-8c84-1af02a5f4172
md"## Parameters"

# ╔═╡ c194b0aa-1ecc-40e8-9193-e6147198bfc2
k = 2.0

# ╔═╡ 320885af-a850-4aa5-b8ad-371063acf39b
R = 4.0

# ╔═╡ 5504bcf5-924b-4c7d-8fbd-0458b5b5892b
pml_δ = 1.0

# ╔═╡ 5f59a261-e21a-4d1d-857a-429cc220f6c2
δ_pml = pml_δ

# ╔═╡ af5bb610-1c5d-479e-a7d2-318d134ccdd2
resolution = 10

# ╔═╡ fc8b07d5-d035-45c3-ac0b-e7ea29f42bf0
N_θ = 1*resolution

# ╔═╡ 5810d1fe-0355-466a-a917-4b590042d364
N_r = 3*resolution

# ╔═╡ 8ac284f0-7116-411a-aeb9-09061e74422f
N_pml = 3*resolution

# ╔═╡ af73f242-133c-41cd-883a-4a4a3cba52a0
cylinder_radius = 1.0

# ╔═╡ 713a5e7f-2200-466e-b5bc-6a18746e4a3e
dim = 2

# ╔═╡ c3b9a58a-f191-4a0d-8e1d-dc595de23421
n_h = 0

# ╔═╡ f25ba6a8-cd2f-4f19-8203-979229345386
function u_ana(x::Vec{2, T}) where {T}
    r, θ = x[1], x[2]
    return hankelh1(n_h, k*r) * exp(im*n_h*θ)
end

# ╔═╡ 6ba2449f-ad24-44d0-a551-38f7c1d88f6b
u_ana(x::Vec{2}, _t::Number) = u_ana(x)

# ╔═╡ 70efa29b-880c-4ac5-b9ef-4f3a84af9ab2
u_ana(n::Node) = u_ana(n.x)

# ╔═╡ 3e1884b6-271d-4abe-88df-2d37cf28a4c4
pml_geom = AnnularPML(R, δ_pml)

# ╔═╡ 314553e9-5c69-4da3-97b0-6287baf20e82
# pml = InvHankelPML(;R, δ=δ_pml, k, m=n_h)
pml = SFB(pml_geom, k)

# ╔═╡ 608866e5-c10e-4ac1-a120-303869e95c31
in_pml(x::Vec{2}) = x[1] >= R

# ╔═╡ 37eb008f-65fb-42b1-8bd2-14827c6e1e68
in_pml(n::Node) = in_pml(n.x)

# ╔═╡ 97120304-9920-42f3-a0b4-0ee6f71457f7
ip = Lagrange{dim, RefCube, 2}()

# ╔═╡ 768d2be4-85e3-4015-9955-fa1682c8fd90
qr = QuadratureRule{dim, RefCube}(4)

# ╔═╡ d688875d-12f6-491a-89a2-1c346dc26acf
pml_qr = QuadratureRule{dim, RefCube}(8)

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

# ╔═╡ 30826935-619b-460a-afc2-872950052f7a
md"## Assembly"

# ╔═╡ 82e06944-aab0-4486-adc5-7b6b9c7d1275
function J_pml(r, θ)
	rθ = PolarCoordinates(r, θ)
	tr, j = tr_and_jacobian(pml, rθ)
	return tr, Tensors.Tensor{2,2,ComplexF64}(j)
end


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

# ╔═╡ eec574a2-f6d0-479c-a47e-11f18dba82da
tv = Tensors.Vec{2, Float64}[Tensors.Vec{2, Float64}((1.7, 0.0)), Tensors.Vec{2, Float64}(([1.8, 0.0]))]

# ╔═╡ b9801fdf-6d31-48a1-9837-219199e0f326
md"## Error measure"

# ╔═╡ 668229dd-ee6e-4d3d-ad87-d3363ffbff47
function integrate_solution(f::Function, u::Vector, cellvalues::CellScalarValues{dim}, dh::DofHandler)  where {dim}
    T = dof_type(dh)

    n_basefuncs = getnbasefunctions(cellvalues)
    global_dofs = zeros(Int, ndofs_per_cell(dh))

	integral = zero(f(first(u), first(dh.grid.nodes).x))

    @inbounds for (cellcount, cell) in enumerate(CellIterator(dh))
        coords = getcoordinates(cell)

        reinit!(cellvalues, cell)
        celldofs!(global_dofs, cell)
		u_local = u[global_dofs]
		for q_point in 1:getnquadpoints(cellvalues)
            dΩ = getdetJdV(cellvalues, q_point)
            coords_qp = spatial_coordinate(cellvalues, q_point, coords)
            r = coords_qp[1]
            θ = coords_qp[2]

			# How to get u?
			# Should we use the PointEvalIterator? Or would interating over that be O(n^2)
			# We could use the get global dofs function and construct manually using shape function?

			# Turn this into a function so we can do a sum reduction rather than a loop?
			u_qp = zero(dof_type(dh))
			for j in 1:n_basefuncs
				u_qp += u_local[j] * shape_value(cellvalues, q_point, j)
			end

			integral += f(u_qp, coords_qp) * r * dΩ
        end

    end
    return integral
end

# ╔═╡ 27d3071a-722e-4c0d-98b9-cd04d7c7980a
md"# Appendix"

# ╔═╡ f9ac9aff-6a1c-41af-8389-b5d6dda301a3
md"## Periodic Constraints"

# ╔═╡ a8c8a3f2-64a9-4f88-abb9-fef16629fc01
function get_nodes_on_face(dh::DofHandler, fi::FaceIndex)
    (cellidx, faceidx) = fi
    _celldofs = celldofs!(dh, cellidx) # extract the dofs for this cell
    r = local_face_dofs_offset[faceidx]:(local_face_dofs_offset[faceidx+1]-1)
    _celldofs[local_face_dofs[r]]
end

# ╔═╡ 9186fcfa-dc52-4d1d-b78a-5cc427cce54b
# Extend to covr dh with multiple fields
function nodedof(dh::DofHandler, cell_idx::Int, node_idx)
    # @assert isclosed(dh)
    return dh.cell_dofs[dh.cell_dofs_offset[cell_idx] + node_idx - 1]
end

# ╔═╡ ccb73afd-3fad-421f-a53a-7abd9b376835
Base.@kwdef mutable struct PeriodicNodeMatch
    # Coordinate along the face for the two nodes
    face_coord::Float64
    # Index of the degree of freedom which we will constrain to be equal to the free dof
    constrained_dof_index::Int = -1
    # Index of the degree of freedom which will set the value of the constrined dof
    free_dof_index::Int = -1
end

# ╔═╡ 0de42268-ee05-45fa-89d3-e7f211382eff
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

# ╔═╡ 9a99bf43-380d-4893-97a3-6d633910f689
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

# ╔═╡ a483d040-f041-4644-9960-c96ac7c4edc6
function node_indices(interpolation, fi::FaceIndex)
    face_idx = fi.idx[2]
    return Ferrite.faces(interpolation)[face_idx]
end

# ╔═╡ 64f6f318-d9ea-444b-8b68-a15895619335
getnode(grid, cell_idx, node_idx) = grid.nodes[grid.cells[cell_idx].nodes[node_idx]]

# ╔═╡ 8087be43-3dc7-4a94-9a47-5ce5e2dcf5aa
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
                coord = getnode(ch.dh.grid, cell_idx, node_idx).x
                free_dof_index = nodedof(ch.dh, cell_idx, node_idx)
                insert_free!(matches, global_to_face_coord(coord), free_dof_index)
            end
        end

        for constrained_face in constrained_faceset
            cell_idx = constrained_face.idx[1]
            for node_idx in node_indices(interpolation, constrained_face)
                coord = getnode(ch.dh.grid, cell_idx, node_idx).x
                constrained_dof_index = nodedof(ch.dh, cell_idx, node_idx)
                insert_constrained!(matches, global_to_face_coord(coord), constrained_dof_index)
            end
        end

		# Remove the corners because they seem to break the apply! function if applied to the same node as a Dirichlet condition. They shouldn't really though...
		face_coord_extrema = extrema(m->m.face_coord, matches)
		filter!(m->m.face_coord ∉ face_coord_extrema, matches)

        for match in matches
            if match.constrained_dof_index < 1 || match.free_dof_index < 1
                warning("Invalid dof index for ", match)
            end
            add!(ch, AffineConstraint(match.constrained_dof_index, [match.free_dof_index=>1.0], 0.0))
        end
    end
    return ch
end

# ╔═╡ 424b9b91-9926-4c54-9f9d-329c974a8336
md"## Grid generation"

# ╔═╡ 0a6cf79e-d111-4f89-b3be-9e2ffc9b5558
equality_constraint(free_node_id::Int, constrained_node_id::Int) = AffineConstraint(constrained_node_id, [free_node_id=>1.0+0.0im], 0.0im)

# ╔═╡ 2eadb933-6d6f-4029-be05-a0df950fd705
md"## Plot utils"

# ╔═╡ e06b65c2-0300-4bf5-8aa7-0784eef693e5
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

# ╔═╡ b02e42a2-a1e0-46be-b70e-c4d0212c2f14
function dof_to_coord(dh::DofHandler)
    global_dofs = zeros(Int, ndofs_per_cell(dh))
    dof_to_node = Vector{Vec{2, Float64}}(undef, dh.ndofs.x)
    for cell in CellIterator(dh)
        coords = getcoordinates(cell)
        celldofs!(global_dofs, cell)
        for (global_dof, coord) in zip(global_dofs, coords)
            dof_to_node[global_dof] = coord
        end
    end
    dof_to_node
end

# ╔═╡ 546e64db-90cb-44a6-a22b-b59791ba36f7
function plot_dof_numbers!(dh::DofHandler)
    for (dof, node) in dof_to_coord_dict(dh)
        GLMakie.text!([string(dof)], position = [(node[1], node[2])])
    end
end

# ╔═╡ 9cfd1f97-f333-40a7-98db-cb9373a857a0
md"## Utils"

# ╔═╡ c323e48f-d5ea-4a4f-8827-b3fc6e163cd0
skipfirst(v) = v[begin+1:end]

# ╔═╡ ab4ff9e7-e92c-4319-bb31-adea87fb16e5
flatten(v) = reshape(v,:)

# ╔═╡ 1071b515-29e3-4abf-a4c9-33890d37f571
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

# ╔═╡ 7b7a469a-01a4-4bd4-8d8a-d35f4db70a3c
dh = let
	dh = DofHandler(ComplexF64, grid)
	push!(dh, :u, 1)
	close!(dh)
	dh
end

# ╔═╡ 05f429a6-ac35-4a90-a45e-8803c7f1692a
u = let
	ch = setup_constraint_handler(dh, u_ana, (x,t)->zero(ComplexF64))
	
	# We should be able to remove this at some point
	K = create_sparsity_pattern(dh, ch)
	K, f = doassemble(cellvalues, pml_cellvalues, K, dh)
	# Or should AffineConstraints be efined in terms of dof number not node id?
    apply!(K, f, ch)
	u = K \ f
	apply!(u, ch)
	u
end

# ╔═╡ 1dfc1298-18b7-4c54-b1a1-fa598b13f527
let
	vtkfile = vtk_grid("helmholtz", dh)
	vtk_point_data(vtkfile, dh, real.(u), "_real")
	vtk_point_data(vtkfile, dh, imag.(u), "_imag")
	# Can't seem to get these to output the cells in the right order...
	dof_coords = dof_to_coord(dh)
    u_ana_nodes = u_ana.(dof_coords)
    vtk_point_data(vtkfile, dh, first.(dof_coords), "_r")
    vtk_point_data(vtkfile, dh, real.(u_ana_nodes), "_exact_real")
	vtk_point_data(vtkfile, dh, imag.(u_ana_nodes), "_exact_imag")
	vtk_save(vtkfile)
end

# ╔═╡ a2c59a93-ed50-44ea-9e7b-8cb99b1f3afb
function plot_u(fnc=real)
    plotter = FerriteViz.MakiePlotter(dh,fnc.(u))
    FerriteViz.surface(plotter,field=:u)
end

# ╔═╡ 0e84a686-6e8f-456c-88c7-1af049e4da96
abs_sq_error = integrate_solution((u,x)-> in_pml(x) ? 0.0 : abs(u - u_ana(x))^2, u, cellvalues, dh)

# ╔═╡ 65b019eb-d01b-4810-991c-59ce48cfe416
abs_sq_norm = integrate_solution((u,x)-> in_pml(x) ? 0.0 : abs(u)^2, u, cellvalues, dh)

# ╔═╡ 339eba6b-be42-4691-a402-ae7ca53e17c6
rel_error = sqrt(abs_sq_error/abs_sq_norm)

# ╔═╡ 291e754a-68d4-42fc-a19e-5ae7bdaa64bb
dof_coords = dof_to_coord(dh)

# ╔═╡ 291dbeb9-42f7-403b-8225-90f2ae98952a
function plot_dof_numbers(dh::DofHandler)
	FerriteViz.wireframe(grid,markersize=5,strokewidth=1,celllabels=true,facelabels=true)
	plot_dof_numbers!(dh)
end

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
# ╠═5504bcf5-924b-4c7d-8fbd-0458b5b5892b
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
# ╠═3e1884b6-271d-4abe-88df-2d37cf28a4c4
# ╠═314553e9-5c69-4da3-97b0-6287baf20e82
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
# ╠═82e06944-aab0-4486-adc5-7b6b9c7d1275
# ╠═baca19e6-ee39-479e-9bd5-f5838cc9f869
# ╟─8296d8a7-41ce-4111-9564-05e7cdc4bfe8
# ╠═eec574a2-f6d0-479c-a47e-11f18dba82da
# ╠═7b7a469a-01a4-4bd4-8d8a-d35f4db70a3c
# ╠═05f429a6-ac35-4a90-a45e-8803c7f1692a
# ╠═1dfc1298-18b7-4c54-b1a1-fa598b13f527
# ╠═a2c59a93-ed50-44ea-9e7b-8cb99b1f3afb
# ╟─b9801fdf-6d31-48a1-9837-219199e0f326
# ╠═668229dd-ee6e-4d3d-ad87-d3363ffbff47
# ╠═0e84a686-6e8f-456c-88c7-1af049e4da96
# ╠═65b019eb-d01b-4810-991c-59ce48cfe416
# ╠═339eba6b-be42-4691-a402-ae7ca53e17c6
# ╟─27d3071a-722e-4c0d-98b9-cd04d7c7980a
# ╟─f9ac9aff-6a1c-41af-8389-b5d6dda301a3
# ╠═a8c8a3f2-64a9-4f88-abb9-fef16629fc01
# ╠═9186fcfa-dc52-4d1d-b78a-5cc427cce54b
# ╠═ccb73afd-3fad-421f-a53a-7abd9b376835
# ╠═0de42268-ee05-45fa-89d3-e7f211382eff
# ╠═9a99bf43-380d-4893-97a3-6d633910f689
# ╠═a483d040-f041-4644-9960-c96ac7c4edc6
# ╠═64f6f318-d9ea-444b-8b68-a15895619335
# ╠═8087be43-3dc7-4a94-9a47-5ce5e2dcf5aa
# ╟─424b9b91-9926-4c54-9f9d-329c974a8336
# ╠═1071b515-29e3-4abf-a4c9-33890d37f571
# ╠═0a6cf79e-d111-4f89-b3be-9e2ffc9b5558
# ╟─2eadb933-6d6f-4029-be05-a0df950fd705
# ╠═291e754a-68d4-42fc-a19e-5ae7bdaa64bb
# ╠═e06b65c2-0300-4bf5-8aa7-0784eef693e5
# ╠═b02e42a2-a1e0-46be-b70e-c4d0212c2f14
# ╠═546e64db-90cb-44a6-a22b-b59791ba36f7
# ╠═291dbeb9-42f7-403b-8225-90f2ae98952a
# ╟─9cfd1f97-f333-40a7-98db-cb9373a857a0
# ╠═c323e48f-d5ea-4a4f-8827-b3fc6e163cd0
# ╠═ab4ff9e7-e92c-4319-bb31-adea87fb16e5
# ╠═b5b526dd-37ef-484c-97fd-2305f0d1d714
# ╠═059eceb1-39fd-4a5e-a787-23d2bbf9b547
# ╟─45be7b0b-c64d-4cfe-a399-de62e6b9094e
# ╠═a51bf9fd-11a6-48ba-9b0d-ab4397be015c
# ╠═c80b6d31-ba85-4d8a-bc28-20dde3f74985
# ╠═b3e4776d-8d03-47c2-8057-8b6e4f443b90
