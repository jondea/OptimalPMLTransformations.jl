
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
    import Test: @test
    using Glob
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

function solve_and_save(;k, N_θ, N_r, N_pml, cylinder_radius=1.0, R=2.0, δ_pml=1.0, u_ana, u_name, order=2, folder, do_optimal)

	nqr_1d = 2*(order + 1)
	bulk_qr=QuadratureRule{2, RefCube}(nqr_1d)
	pml_qr=QuadratureRule{2, RefCube}(nqr_1d)

    result_folder="$folder/k_$k/n_pml_$N_pml/u_$u_name"
	run(`mkdir -p $result_folder`)

	# SFB
	let
		pml = SFB(AnnularPML(R, δ_pml), k)
        params = PMLHelmholtzPolarAnnulusParams(;k, N_θ, N_r, N_pml, cylinder_radius, R, δ_pml, u_ana, order, pml, bulk_qr, pml_qr)
		result = solve(params)
        write_csv_line(result_folder, params, result, u_name, "SFB", "GL$(nqr_1d)x$(nqr_1d)")
	end

    # InvHankel 0 as a not-necessarily-optimal PML
    let
        pml = InvHankelPML(;R, δ=δ_pml, k, m=0)
        params = PMLHelmholtzPolarAnnulusParams(;k, N_θ, N_r, N_pml, cylinder_radius, R, δ_pml, u_ana, order, pml, bulk_qr, pml_qr)
        result = solve(params)
        write_csv_line(result_folder, params, result, u_name, "InvHankel0", "GL$(nqr_1d)x$(nqr_1d)")
    end

    # If u_ana is a single hankel mode, then the InvHankelPML is optimal. Test that this is true
    if typeof(u_ana) <: SingleAngularFourierMode
        n_h = u_ana.m
        # InvHankel n_h with increasing N_pml. This is the standard PML approach. If n_h == 0, then we have already done this
        if n_h != 0
            pml = InvHankelPML(;R, δ=δ_pml, k, m=n_h)
            params = PMLHelmholtzPolarAnnulusParams(;k, N_θ, N_r, N_pml, cylinder_radius, R, δ_pml, u_ana, order, pml, bulk_qr, pml_qr)
            result = solve(params)
            write_csv_line(result_folder, params, result, u_name, "InvHankel$n_h", "GL$(nqr_1d)x$(nqr_1d)")
        end

        # InvHankel n_h with N_pml=1 and increasing integration order. This is a bit like the optimal PML approach
        let
            pml = InvHankelPML(;R, δ=δ_pml, k, m=n_h)
            aniso_qr=anisotropic_quadrature(RefCube, N_pml*nqr_1d, nqr_1d)
            params = PMLHelmholtzPolarAnnulusParams(;k, N_θ, N_r, N_pml=1, cylinder_radius, R, δ_pml, u_ana, order, pml, bulk_qr, pml_qr=aniso_qr)
            result = solve(params)
            write_csv_line(result_folder, params, result, u_name, "InvHankel$n_h", "GL$(N_pml*nqr_1d)x$(nqr_1d)")
        end
    end

    if do_optimal
        pml = InvHankelSeriesPML(AnnularPML(R, δ_pml), HankelSeries(u_ana))
        params = PMLHelmholtzPolarAnnulusParams(;k, N_θ, N_r, N_pml=0, cylinder_radius, R, δ_pml, u_ana, order, pml, bulk_qr, pml_qr)
        result = solve(params)
        write_csv_line(result_folder, params, result, u_name, "Optimal", "Optimal")
    end
end

function run_all(;full_run=false, cylinder_radius=1.0, R=2.0)
	folder="study_" * Dates.format(now(), dateformat"YYYY-mm-dd_HH-MM-SS")

	if full_run
		resolutions = [1, 2, 4, 8, 16, 32]
		N_pmls = [1, 2, 4, 8, 16, 32]
	else
		resolutions = [8]
		N_pmls = [1, 4]
	end
	ks = [0.1, 1.0, 10.0]

    @showprogress for res in resolutions, k in ks, N_pml in N_pmls
        u_anas = [
            "0" => single_hankel_mode(k, 0, 1/hankelh1(0, k*cylinder_radius)),
            "3" => single_hankel_mode(k, 3, 1/hankelh1(3, k*cylinder_radius)),
            "scattered" => HankelSeries(k, scattered_coef(-10:10, k)),
        ]
        N_θ = round(Int, τ * max(res, k*res))
        N_r = round(Int, (R - cylinder_radius) * max(res, k*res))
        for (u_name, u_ana) in u_anas
            do_optimal = N_pml == maximum(N_pmls)
            solve_and_save(;k, N_θ, N_r, u_name, u_ana, N_pml, folder, do_optimal, cylinder_radius, R)
        end
    end

    # Collate individual results into one csv
	write("$folder/result.csv", "k,u_name,N_θ,N_r,N_pml,pml,integration,assemble_time,solve_time,abs_sq_error,abs_sq_norm,rel_error\n")
	for filename in glob(glob"*/*/*/result.csv", folder)
		open("$folder/result.csv", "a") do outfile
			write(outfile, read(filename))
		end
	end

    CSV.read("$folder/result.csv", DataFrame)
end

results_df = run_all(;full_run=("--full_run" in ARGS))

if "--test_run" in ARGS
    validata = CSV.read("validata/result.csv", DataFrame)

    df = leftjoin(validata, results_df; on=[:k, :u_name, :N_θ, :N_r, :N_pml, :pml, :integration], renamecols="_exact"=>"_test")
    df.assemble_time_ratio = df.assemble_time_test ./ df.assemble_time_exact
    df.solve_time_ratio = df.solve_time_test ./ df.solve_time_exact
    df.abs_sq_error_ratio = df.abs_sq_error_test ./ df.abs_sq_error_exact
    df.abs_sq_norm_ratio = df.abs_sq_norm_test ./ df.abs_sq_norm_exact
    select!(df, [:k, :n_h, :N_θ, :N_r, :N_pml, :pml, :integration, :abs_sq_error_ratio, :abs_sq_norm_ratio, :assemble_time_ratio, :solve_time_ratio])

    @test all(df.abs_sq_norm_ratio .≈ 1)

    println("Improved")
    display(sort(filter(:abs_sq_error_ratio => <(0.999), df), :abs_sq_error_ratio))
    println("Got worse")
    display(sort(filter(:abs_sq_error_ratio => >(1.001), df), :abs_sq_error_ratio; rev=true))

    @test all(df.abs_sq_error_ratio .< 1.05)
end

