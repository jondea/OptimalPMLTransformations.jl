module OptimalPMLTransformations

import SpecialFunctions: hankelh1
import StaticArrays: SVector
import ProgressMeter: @showprogress
import OffsetArrays: OffsetVector

export PMLGeometry
export PMLCoordinates
export AnnularPML,
       XAlignedRectangularPML,
       ConformalPML
export polar_to_pml_coordinates,
       pml_to_polar_coordinates
export polar_to_pml_coordinates,
       pml_to_polar_coordinates

export optimal_pml_transformation

export AbstractFieldFunction

export PlanarWave

export Rip2D, classify_outer_boundary, find_rips


include("pml_geometries/pml_geometries.jl")

include("fields/fields.jl")

include("transformations/transformations.jl")

include("integration/integration.jl")

include("rips.jl")

end # module
