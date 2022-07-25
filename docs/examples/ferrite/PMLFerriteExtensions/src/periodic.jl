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


# Extend to covr dh with multiple fields
function nodedof(dh::DofHandler, cell_idx::Int, node_idx)
    # @assert isclosed(dh)
    return dh.cell_dofs[dh.cell_dofs_offset[cell_idx] + node_idx - 1]
end

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
