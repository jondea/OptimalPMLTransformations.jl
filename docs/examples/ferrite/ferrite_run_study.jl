
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
    using Dates
end

@revise using Ferrite

@revise using OptimalPMLTransformations

@revise using PMLFerriteExtensions

include("pml_helmholtz_polar_annulus_assembly.jl")

function write_csv_line(folder::String, params::PMLHelmholtzPolarAnnulusParams, result::Result, u_name::String, pml_name::String, integration_scheme::String)
    @unpack k, N_θ, N_r, N_pml, cylinder_radius, R, δ_pml, u_ana, order = params
    @unpack assemble_time, solve_time, abs_sq_error, abs_sq_norm, rel_error = result
    open("$folder/result.csv","a") do f
        println(f,"$k,$u_name,$N_θ,$N_r,$N_pml,$pml_name,$integration_scheme,$assemble_time,$solve_time,$abs_sq_error,$abs_sq_norm,$rel_error")
    end
end

function solve_with_catch()
    result = Result()
    try
        result = solve(params)
    catch e
        @error e
    end
    return result
end

function solve_and_save(;k, N_θ, N_r, N_pml, cylinder_radius=1.0, R=2.0, δ_pml=1.0, n_h, order=2, folder)

	nqr_1d = 2*(order + 1)
	bulk_qr=QuadratureRule{2, RefCube}(nqr_1d)
	pml_qr=QuadratureRule{2, RefCube}(nqr_1d)
	u_ana=single_hankel_mode(k, n_h)
    u_name=string(n_h)

    result_folder="$folder/k_$k/n_pml_$N_pml/n_h_$n_h"
	run(`mkdir -p $result_folder`)

	# SFB
	let
		pml = SFB(AnnularPML(R, δ_pml), k)
        params = PMLHelmholtzPolarAnnulusParams(;k, N_θ, N_r, N_pml, cylinder_radius, R, δ_pml, u_ana, order, pml, bulk_qr, pml_qr)
		result = solve(params)
        write_csv_line(result_folder, params, result, u_name, "SFB", "GL$(nqr_1d)x$(nqr_1d)")
	end

	# InvHankel n_h
	let
		pml = InvHankelPML(;R, δ=δ_pml, k, m=n_h)
        params = PMLHelmholtzPolarAnnulusParams(;k, N_θ, N_r, N_pml, cylinder_radius, R, δ_pml, u_ana, order, pml, bulk_qr, pml_qr)
		result = solve(params)
        write_csv_line(result_folder, params, result, u_name, "InvHankel$n_h", "GL$(nqr_1d)x$(nqr_1d)")
	end

	# InvHankel 0
    if n_h != 0
        pml = InvHankelPML(;R, δ=δ_pml, k, m=0)
        params = PMLHelmholtzPolarAnnulusParams(;k, N_θ, N_r, N_pml, cylinder_radius, R, δ_pml, u_ana, order, pml, bulk_qr, pml_qr)
		result = solve(params)
        write_csv_line(result_folder, params, result, u_name, "InvHankel0", "GL$(nqr_1d)x$(nqr_1d)")
	end

	# InvHankel n_h with N_pml=1 and increasing integration order
	let
		pml = InvHankelPML(;R, δ=δ_pml, k, m=n_h)
		aniso_qr=anisotropic_quadrature(RefCube, N_pml*nqr_1d, nqr_1d)
        params = PMLHelmholtzPolarAnnulusParams(;k, N_θ, N_r, N_pml=1, cylinder_radius, R, δ_pml, u_ana, order, pml, bulk_qr, pml_qr=aniso_qr)
		result = solve(params)
        write_csv_line(result_folder, params, result, u_name, "InvHankel$n_h", "GL$(N_pml*nqr_1d)x$(nqr_1d)")
	end
end

function run_all(;full_run=false)
	folder="study_" * Dates.format(now(), dateformat"YYYY-mm-dd_HH-MM-SS")

	if full_run
		resolutions = [1, 2, 4, 8, 16, 32]
		N_pmls = [1, 2, 4, 8, 16, 32]
	else
		resolutions = [2, 8]
		N_pmls = [1, 4]
	end
	n_hs = [0, 3]
	ks = [0.1, 1.0, 10.0]
	@showprogress [solve_and_save(;k, N_θ=max(n_h, 1)*res, N_r=round(Int, max(res, k*res)), n_h, N_pml, folder) for res in resolutions, n_h in n_hs, k in ks, N_pml in N_pmls]

    # Collate individual results into one csv
	write("$folder/result.csv", "k,n_h,N_θ,N_r,N_pml,pml,integration,assemble_time,solve_time,abs_sq_error,abs_sq_norm,rel_error\n")
	for n_h in n_hs, k in ks, N_pml in N_pmls
		result_folder="$folder/k_$k/n_pml_$N_pml/n_h_$n_h"
		open("$folder/result.csv", "a") do outfile
			write(outfile, read("$result_folder/result.csv"))
		end
	end

    CSV.read("$folder/result.csv", DataFrame)
end

results_df = run_all(;full_run=("--full_run" ∈ ARGS))
