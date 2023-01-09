module OptimalPMLTransformations

import SpecialFunctions: hankelh1, besselj
import StaticArrays: SVector, SMatrix, @SMatrix, SDiagonal, SA
import ProgressMeter: @showprogress
import OffsetArrays: OffsetVector
import LinearAlgebra: I, norm
import Einsum: @einsum
import InverseHankelFunction: invhankelh1n, diffinvhankelh1n, diffhankelh1
import CubicHermiteSpline
import CubicHermiteSpline: CubicHermiteSplineInterpolation
import FastGaussQuadrature: gausslegendre
import IterTools: partition

"Tensor contraction of two vectors"
contract(x::AbstractVector, y::AbstractVector) = mapreduce(*, +, x, y)

export insertsorted!

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

export jacobian
export tν
export tν_jacobian
export tν_and_jacobian
export tx
export tx_jacobian
export tx_and_jacobian
export tr
export tr_jacobian
export tr_and_jacobian

export SFB

export AbstractFieldFunction
export PMLFieldFunction

export ∂u_∂tr

export PlanarWave

export PML

export SingleAngularFourierMode,
       single_hankel_mode,
       single_angular_fourier_mode
export InvHankelPML

export HankelSeries,
       two_mode_pole_coef,
       two_mode_pole_series,
       scattered_coef

export InvHankelSeriesPML
export add_interpolation!
export has_interpolation

export interpolate,
       InterpPoint,
       InterpLine,
       ContinuousInterpolation,
       Interpolation,
       InterpPatch,
       eachpatch,
       Dtν_ν,
       Dtν_νζ,
       ∂tν_∂ν,
       refine!,
       refine_in_ζ!,
       refine,
       evaluate,
       evalute_and_correct
export continue_in_ζ
export Rip
export eval_hermite_patch

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

export integrate
export integrate_hcubature
export integrate_between
export integrate_quad

export optimal_pml_transformation_solve

include("utils.jl")

include("coordinates/coordinates.jl")

include("pml_geometries/pml_geometries.jl")

include("fields/fields.jl")

include("interpolation/interpolation.jl")

include("transformations/transformations.jl")

include("rips/rips.jl")

include("integration/integration.jl")

end # module
