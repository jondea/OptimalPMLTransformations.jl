skipfirst(v) = v[begin+1:end]

flatten(v) = reshape(v,:)

function Ferrite.DofHandler(T::DataType, grid::Ferrite.AbstractGrid, fields::Array{Symbol})
	dh = DofHandler(T, grid)
	for field in fields
		push!(dh, field, 1)
	end
	close!(dh)
	dh
end


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
