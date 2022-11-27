
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

function solve_and_save(;k, N_θ, N_r, N_pml, cylinder_radius=1.0, R=2.0, δ_pml=1.0, n_h, order=2, folder)

	nqr_1d = 2*(order + 1)
	qr=QuadratureRule{2, RefCube}(nqr_1d)
	pml_qr=QuadratureRule{2, RefCube}(nqr_1d)
	u_ana=single_hankel_mode(k,n_h)

	result_folder="$folder/k_$k/n_pml_$N_pml/n_h_$n_h"
	run(`mkdir -p $result_folder`)

	# SFB
	let
		pml = SFB(AnnularPML(R, δ_pml), k)
		(assemble_time, solve_time, abs_sq_error, abs_sq_norm, rel_error) = solve_for_error(;k, N_θ, N_r, N_pml, cylinder_radius, R, δ_pml, u_ana, order, pml, qr, pml_qr)
		open("$result_folder/result.csv","a") do f
			println(f, "$k,$n_h,$N_θ,$N_r,$N_pml,SFB,GL$(nqr_1d)x$(nqr_1d),$assemble_time,$solve_time,$abs_sq_error,$abs_sq_norm,$rel_error")
		end
	end

	# InvHankel n_h
	let
		pml = InvHankelPML(;R, δ=δ_pml, k, m=n_h)
		(assemble_time, solve_time, abs_sq_error, abs_sq_norm, rel_error) = solve_for_error(;k, N_θ, N_r, N_pml, cylinder_radius, R, δ_pml, u_ana, order, pml, qr, pml_qr)
		open("$result_folder/result.csv","a") do f
			println(f,"$k,$n_h,$N_θ,$N_r,$N_pml,InvHankel$n_h,GL$(nqr_1d)x$(nqr_1d),$assemble_time,$solve_time,$abs_sq_error,$abs_sq_norm,$rel_error")
		end
	end

	# InvHankel 0
    if n_h != 0
		pml = InvHankelPML(;R, δ=δ_pml, k, m=0)
		(assemble_time, solve_time, abs_sq_error, abs_sq_norm, rel_error) = solve_for_error(;k, N_θ, N_r, N_pml, cylinder_radius, R, δ_pml, u_ana, order, pml, qr, pml_qr)
		open("$result_folder/result.csv","a") do f
			println(f, "$k,$n_h,$N_θ,$N_r,$N_pml,InvHankel0,GL$(nqr_1d)x$(nqr_1d),$assemble_time,$solve_time,$abs_sq_error,$abs_sq_norm,$rel_error")
		end
	end

	# InvHankel n_h with N_pml=1 and increasing integration order
	let
		pml = InvHankelPML(;R, δ=δ_pml, k, m=n_h)
		aniso_pml_qr=anisotropic_quadrature(RefCube, N_pml*nqr_1d, nqr_1d)
		(assemble_time, solve_time, abs_sq_error, abs_sq_norm, rel_error) = solve_for_error(;k, N_θ, N_r, N_pml=1, cylinder_radius, R, δ_pml, u_ana, order, pml, qr, pml_qr=aniso_pml_qr)
		open("$result_folder/result.csv","a") do f
			println(f,"$k,$n_h,$N_θ,$N_r,1,InvHankel$n_h,GL$(N_pml*nqr_1d)x$(nqr_1d),$assemble_time,$solve_time,$abs_sq_error,$abs_sq_norm,$rel_error")
		end
	end
end

function run_all(;test_run=false)
	folder=tempname("./")

	if test_run
		resolutions = [2, 8]
		N_pmls = [1, 4]
	else
		resolutions = [1, 2, 4, 8, 16, 32]
		N_pmls = [1, 2, 4, 8, 16, 32]
	end
	n_hs = [0, 3]
	ks = [0.1, 1.0, 10.0]
	@showprogress [solve_and_save(;k, N_θ=max(n_h, 1)*res, N_r=round(Int, max(res, k*res)), n_h, N_pml, folder) for res in resolutions, n_h in n_hs, k in ks, N_pml in N_pmls]

	write("$folder/result.csv", "k,n_h,N_θ,N_r,N_pml,pml,integration,assemble_time,solve_time,abs_sq_error,abs_sq_norm,rel_error\n")
	for res in resolutions, n_h in n_hs, k in ks, N_pml in N_pmls
		result_folder="$folder/k_$k/n_pml_$N_pml/n_h_$n_h"
		open("$folder/result.csv", "a") do outfile
			write(outfile, read("$result_folder/result.csv"))
		end
	end

    CSV.read("$folder/result.csv", DataFrame)
end

# results_df = run_all(test_run=true)

# filter(r->r.N_pml == 1, results_df);
