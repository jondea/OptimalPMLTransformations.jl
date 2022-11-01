

# Modify this to be generic in order, with a stretching and hard code the periodicity
# QuadraticQuadrilateral
function generate_pml_grid(::Type{QuadraticQuadrilateral}, N_θ, N_r, N_pml, r_inner, R, δ_pml)
    ncell_x = (N_r + N_pml)
 	ncell_y = N_θ
    nodes_x = collect(range(r_inner, R, length=2*N_r+1))
    if N_pml != 0
        append!(nodes_x, skipfirst(range(R, R+δ_pml, length=2*N_pml+1)))
    end
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
    onright(x) = x[1]≈(R + (N_pml>0)*δ_pml)

    addfaceset!(grid, "bottom", onbottom)
    addfaceset!(grid, "right", onright)
    addfaceset!(grid, "top", ontop)
    addfaceset!(grid, "left", onleft)

    addnodeset!(grid, "bottom", onbottom)
    addnodeset!(grid, "right", onright)
    addnodeset!(grid, "top", ontop)
    addnodeset!(grid, "left", onleft)

    return grid
end
