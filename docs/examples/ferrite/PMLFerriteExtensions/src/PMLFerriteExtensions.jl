module PMLFerriteExtensions

using Ferrite
using Tau

include("utils.jl")
include("grid.jl")
include("periodic.jl")
include("quadrature.jl")
include("plot.jl")


export generate_pml_grid
export add_periodic!
export anisotropic_quadrature
export integrate_solution
export dof_to_coord
export plot_dof_numbers!
export write_vtk

end
