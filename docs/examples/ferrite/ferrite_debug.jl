
begin
    import Pkg
    # activate the shared project environment
    Pkg.activate(@__DIR__)
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
	using IterTools
	using OffsetArrays
	using ProgressMeter
	import ProgressMeter: update!
    import FerriteViz, GLMakie
    import CSV
    using DataFrames
end

@revise using Ferrite

@revise using OptimalPMLTransformations

@revise using PMLFerriteExtensions

# Single case for debugging
# solve_and_save(;k=0.1, N_θ=3, N_r=1, N_pml=1, cylinder_radius=1.0, R=2.0, δ_pml=1.0, n_h=3, order=2, folder=tempname("./"))

include("pml_helmholtz_polar_annulus_assembly.jl")

let
    k = 1.0
    n_h = 0
    R = 4.0
    N_θ = 1
    N_pml = 0
    N_r = 1

    N_θ = 10
    N_r = 60

    order = 2
    δ_pml=1.0
    cylinder_radius = 1.0
    nqr_1d = 2*(order + 1)
    qr=QuadratureRule{2, RefCube}(nqr_1d)
    pml_qr=QuadratureRule{2, RefCube}(nqr_1d)
    u_ana=HankelSeries(k, OffsetVector([1.0], n_h:n_h))
    # u_ana=two_mode_pole_series(k, R + (1.0+1.0im))

# Optimal
    pml = InvHankelSeriesPML(AnnularPML(R, δ_pml), u_ana)
    (assemble_time, solve_time, abs_sq_error, abs_sq_norm, rel_error) = solve_for_error(;k, N_θ, N_r, N_pml, cylinder_radius, R, δ_pml, u_ana, order, pml, qr, pml_qr)
    @show (assemble_time, solve_time, abs_sq_error, abs_sq_norm, rel_error)
end
