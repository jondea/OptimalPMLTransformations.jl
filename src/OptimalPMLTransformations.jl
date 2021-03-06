module OptimalPMLTransformations

import SpecialFunctions: hankelh1, besselj
import StaticArrays: SVector, SMatrix, @SMatrix, SDiagonal
import ProgressMeter: @showprogress
import OffsetArrays: OffsetVector
import LinearAlgebra: I
import Einsum: @einsum
import InverseHankelFunction: invhankelh1n, diffinvhankelh1n, diffhankelh1

"Tensor contraction of two vectors"
contract(x::AbstractVector, y::AbstractVector) = mapreduce(*, +, x, y)

export PMLGeometry
export PMLCoordinates
export CartesianCoordinates
export PolarCoordinates
export convert
export AnnularPML,
       XAlignedRectangularPML,
       ConformalPML
export polar_to_pml_coordinates,
       pml_to_polar_coordinates
export polar_to_pml_coordinates,
       pml_to_polar_coordinates

export optimal_pml_transformation

export tν
export tν_jacobian
export tν_and_jacobian
export tx
export tx_jacobian
export tx_and_jacobian
export tr
export tr_jacobian
export tr_and_jacobian

export AbstractFieldFunction

export PlanarWave

export SingleAngularFourierMode,
       single_hankel_mode,
       single_angular_fourier_mode

export HankelSeries,
       two_mode_pole_coef,
       scattered_coef

export Rip2D, classify_outer_boundary, find_rips

export gausslegendreunit,
       int_gauss,
       int_gauss_2d,
       gausslegendreunittrans,
       int_gauss_trans,
       int_gauss_trans_2d,
       gausslegendretrans_mid,
       int_gauss_trans_mid,
       gausslegendretrans_mid,
       int_gauss_trans_2d_mid

export optimal_pml_transformation_solve
include("coordinates/coordinates.jl")

include("pml_geometries/pml_geometries.jl")

include("fields/fields.jl")

include("transformations/transformations.jl")

include("integration/integration.jl")

include("rips/rips.jl")

end # module
