
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

function write_vtk(filename::String, dh::Ferrite.AbstractDofHandler, u::Vector{<:Complex}, u_ana=nothing)
	vtkfile = vtk_grid(filename, dh)
	vtk_point_data(vtkfile, dh, real.(u), "_real")
	vtk_point_data(vtkfile, dh, imag.(u), "_imag")
	if !isnothing(u_ana)
		dof_coords = dof_to_coord(dh)
	    u_ana_nodes = u_ana.(dof_coords)
	    vtk_point_data(vtkfile, dh, real.(u_ana_nodes), "_exact_real")
		vtk_point_data(vtkfile, dh, imag.(u_ana_nodes), "_exact_imag")
	end
	vtk_save(vtkfile)
end

function plot_dof_numbers!(dh::DofHandler)
    for (dof, node) in enumerate(dof_to_coord(dh))
        GLMakie.text!([string(dof)], position = [(node[1], node[2])])
    end
end

function plot_dof_numbers(dh::DofHandler)
	FerriteViz.wireframe(grid,markersize=5,strokewidth=1,celllabels=true,facelabels=true)
	plot_dof_numbers!(dh)
end
