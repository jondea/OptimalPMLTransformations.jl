### A Pluto.jl notebook ###
# v0.19.11

using Markdown
using InteractiveUtils

# ╔═╡ a51bf9fd-11a6-48ba-9b0d-ab4397be015c
begin
    import Pkg
    # activate the shared project environment
    Pkg.activate(Base.current_project())
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
	using ProgressMeter
	import ProgressMeter: update!
    import FerriteViz, GLMakie
    import CSV
    using DataFrames
end

# ╔═╡ c80b6d31-ba85-4d8a-bc28-20dde3f74985
@revise using Ferrite

# ╔═╡ b3e4776d-8d03-47c2-8057-8b6e4f443b90
@revise using OptimalPMLTransformations

# ╔═╡ 424b9b91-9926-4c54-9f9d-329c974a8336
@revise using PMLFerriteExtensions

# ╔═╡ 4c152d80-c523-4da1-9035-7d7b4cb144d9
md"# PML Helmholtz equation in annulus in polar coordinates"

# ╔═╡ 1d9ac613-63c0-4c34-8c84-1af02a5f4172
md"## Parameters"

# ╔═╡ c75d6e53-ee17-4587-925c-f253705b633d
# Single case for debugging
# solve_and_save(;k=0.1, N_θ=3, N_r=1, N_pml=1, cylinder_radius=1.0, R=2.0, δ_pml=1.0, n_h=3, order=2, folder=tempname("./"))

# ╔═╡ 0d050dc4-0b97-4a0c-b399-970e1f7284c9
md"## Constraints"

# ╔═╡ 53ae2bac-59b3-44e6-9d6c-53a18567ea2f
function setup_constraint_handler(dh::Ferrite.AbstractDofHandler, left_bc, right_bc)
	ch = ConstraintHandler(dh)

	gridfaceset(s) = getfaceset(dh.grid, s)

    add_periodic!(ch, [gridfaceset("bottom")=>gridfaceset("top")], x->x[1])

    inner_dbc = Dirichlet(:u, gridfaceset("left"), left_bc)
	add!(ch, inner_dbc)

	outer_dbc = Dirichlet(:u, gridfaceset("right"), right_bc)
	add!(ch, outer_dbc)

    close!(ch)
	Ferrite.update!(ch, 0.0)
	ch
end

# ╔═╡ 30826935-619b-460a-afc2-872950052f7a
md"## Assembly"

# ╔═╡ 8296d8a7-41ce-4111-9564-05e7cdc4bfe8
md"## Solve and plot"

# ╔═╡ 27d3071a-722e-4c0d-98b9-cd04d7c7980a
md"# Appendix"

# ╔═╡ 9cfd1f97-f333-40a7-98db-cb9373a857a0
md"## Utils"

# ╔═╡ b5b526dd-37ef-484c-97fd-2305f0d1d714
html"""
<style>
  main {
    max-width: 900px;
  }
</style>
"""

# ╔═╡ 059eceb1-39fd-4a5e-a787-23d2bbf9b547
PlutoUI.TableOfContents()

# ╔═╡ 759eb280-f138-46f8-af4e-7447a38c4a9f
(f::SingleAngularFourierMode)(c::Node) = f(c.x)

# ╔═╡ 424be202-5f13-46e1-a4db-8cbd6937fdcc
(f::SingleAngularFourierMode)(x::Vec{2}, _t::Number) = f(x)

# ╔═╡ ea07bf0c-552c-46fe-9431-4ca6a99e3647
(f::SingleAngularFourierMode)(x::Vec{2}) = f(PolarCoordinates(x[1],x[2]))

# ╔═╡ 691593a3-4e68-4b3c-8020-ee07bf773486
Base.in(n::Node, pml::PMLGeometry) = in(n.x, pml)

# ╔═╡ 8947ed4c-ac37-4531-9252-63b195b73577
Base.in(x::Vec{2}, pml::AnnularPML) = x[1] >= pml.R

# ╔═╡ 7fca1678-2d50-4ade-be8b-43430f751fcf
Base.in(x, pml::SFB) = in(x, pml.geom)

# ╔═╡ 04a4723c-c9d3-46cd-8d22-48b9c992f366
Base.in(x, pml::InvHankelPML) = in(x, pml.geom)

# ╔═╡ baca19e6-ee39-479e-9bd5-f5838cc9f869
function doassemble(cellvalues::CellScalarValues{dim}, pml_cellvalues::CellScalarValues{dim},
                         K::SparseMatrixCSC, dh::DofHandler, pml, k::Number) where {dim}
    T = dof_type(dh)
	fill!(K.nzval, zero(T))
    f = zeros(T, ndofs(dh))
    assembler = start_assemble(K, f)

    n_basefuncs = getnbasefunctions(cellvalues)
    global_dofs = zeros(Int, ndofs_per_cell(dh))

    fe = zeros(T, n_basefuncs) # Local force vector
    Ke = zeros(T, n_basefuncs, n_basefuncs) # Local stiffness mastrix

    for (cellcount, cell) in enumerate(CellIterator(dh))
        fill!(Ke, 0)
        fill!(fe, 0)
        coords = getcoordinates(cell)
		if mean(coords) ∈ pml
			reinit!(pml_cellvalues, cell)
			for q_point in 1:getnquadpoints(pml_cellvalues)
	            dΩ = getdetJdV(pml_cellvalues, q_point)
	            coords_qp = spatial_coordinate(pml_cellvalues, q_point, coords)
	            r = coords_qp[1]
	            θ = coords_qp[2]
				tr, J_ = tr_and_jacobian(pml, PolarCoordinates(r, θ))
				J_pml = Tensors.Tensor{2,2,ComplexF64}(J_)
	            Jᵣₓ = diagm(Tensor{2,2}, [1.0, 1/tr])
	            Jₜᵣᵣ = inv(J_pml)
				detJₜᵣᵣ	= det(J_pml)
	            for i in 1:n_basefuncs
	                δu = shape_value(pml_cellvalues, q_point, i)
	                ∇δu = shape_gradient(pml_cellvalues, q_point, i)
	                for j in 1:n_basefuncs
	                    u = shape_value(pml_cellvalues, q_point, j)
	                    ∇u = shape_gradient(pml_cellvalues, q_point, j)
	                    Ke[i, j] += ((Jₜᵣᵣ ⋅ (Jᵣₓ ⋅ ∇δu)) ⋅ (Jₜᵣᵣ ⋅ (Jᵣₓ⋅∇u)) - k^2*δu * u
										) * tr * detJₜᵣᵣ * dΩ
	                end
	            end
	        end
		else
			# Bulk
			reinit!(cellvalues, cell)
			for q_point in 1:getnquadpoints(cellvalues)
	            dΩ = getdetJdV(cellvalues, q_point)
				coords_qp = spatial_coordinate(cellvalues, q_point, coords)
	            r = coords_qp[1]
				Jᵣₓ = diagm(Tensor{2,2}, [1.0, 1/r])
	            for i in 1:n_basefuncs
	                δu = shape_value(cellvalues, q_point, i)
	                ∇δu = shape_gradient(cellvalues, q_point, i)
	                for j in 1:n_basefuncs
	                    u = shape_value(cellvalues, q_point, j)
	                    ∇u = shape_gradient(cellvalues, q_point, j)
	                    Ke[i, j] += ((Jᵣₓ ⋅∇δu)⋅(Jᵣₓ ⋅∇u) - k^2*δu * u) * r * dΩ
	                end
	            end
	        end
		end

        celldofs!(global_dofs, cell)
        assemble!(assembler, global_dofs, fe, Ke)
    end
    return K, f
end

# ╔═╡ 5ec5d21a-3bcf-4cde-a814-f4fb71c30a2f
function solve_for_error(;k, N_θ, N_r, N_pml, cylinder_radius=1.0, R=2.0, δ_pml=1.0, u_ana, order=2, pml, qr::QuadratureRule, pml_qr::QuadratureRule)
	# Set to NaN initially in case we throw
	(assemble_time, solve_time, abs_sq_error, abs_sq_norm, rel_error) = Iterators.cycle(NaN)
	try
		dim=2

		ip = Lagrange{dim, RefCube, order}()
		if order == 2
			grid = generate_pml_grid(QuadraticQuadrilateral, N_θ, N_r, N_pml, cylinder_radius, R, δ_pml)
		else
			error()
		end

		dh = DofHandler(ComplexF64, grid, [:u])
		ch = setup_constraint_handler(dh, u_ana, (x,t)->zero(ComplexF64))

		cellvalues = CellScalarValues(qr, ip);
		pml_cellvalues = CellScalarValues(pml_qr, ip);
		K = create_sparsity_pattern(ch.dh, ch)
		assemble_time = @elapsed K, f = doassemble(cellvalues, pml_cellvalues, K, ch.dh, pml, k)
	    apply!(K, f, ch)
		solve_time = @elapsed u = K \ f
		apply!(u, ch)
		abs_sq_error = integrate_solution((u,x)-> x∈pml ? 0.0 : abs(u - u_ana(x))^2, u, cellvalues, dh)
		abs_sq_norm = integrate_solution((u,x)-> x∈pml ? 0.0 : abs(u)^2, u, cellvalues, dh)
		rel_error = sqrt(abs_sq_error/abs_sq_norm)
	catch e
        @error e
	end
	return (assemble_time, solve_time, abs_sq_error, abs_sq_norm, rel_error)
end

# ╔═╡ ef1dac4e-77ea-47ef-a94a-bf63395120bc
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

# ╔═╡ 4770eea5-ddab-4d98-90b9-4f95fec8a23e
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

# ╔═╡ 76070b3b-2663-481f-8deb-cf995bb9b640
results_df = run_all(test_run=true)

# ╔═╡ 1d6054f2-6ae5-4d09-8c7e-fe536512e3c6
filter(r->r.N_pml == 1, results_df);

# ╔═╡ d87c0552-eec5-4765-ad04-03fbab2727fe
function solve(ch, cellvalues, pml_cellvalues)
	K = create_sparsity_pattern(ch.dh, ch)
	K, f = doassemble(cellvalues, pml_cellvalues, K, ch.dh)
    apply!(K, f, ch)
	u = K \ f
	apply!(u, ch)
	return u
end

# ╔═╡ 45be7b0b-c64d-4cfe-a399-de62e6b9094e
md"## Dependencies"

# ╔═╡ Cell order:
# ╟─4c152d80-c523-4da1-9035-7d7b4cb144d9
# ╟─1d9ac613-63c0-4c34-8c84-1af02a5f4172
# ╠═76070b3b-2663-481f-8deb-cf995bb9b640
# ╠═1d6054f2-6ae5-4d09-8c7e-fe536512e3c6
# ╠═c75d6e53-ee17-4587-925c-f253705b633d
# ╠═4770eea5-ddab-4d98-90b9-4f95fec8a23e
# ╠═5ec5d21a-3bcf-4cde-a814-f4fb71c30a2f
# ╠═ef1dac4e-77ea-47ef-a94a-bf63395120bc
# ╟─0d050dc4-0b97-4a0c-b399-970e1f7284c9
# ╠═53ae2bac-59b3-44e6-9d6c-53a18567ea2f
# ╟─30826935-619b-460a-afc2-872950052f7a
# ╠═baca19e6-ee39-479e-9bd5-f5838cc9f869
# ╠═721a855f-49aa-40ae-93ad-a1cc5313e399
# ╟─8296d8a7-41ce-4111-9564-05e7cdc4bfe8
# ╠═d87c0552-eec5-4765-ad04-03fbab2727fe
# ╟─27d3071a-722e-4c0d-98b9-cd04d7c7980a
# ╟─9cfd1f97-f333-40a7-98db-cb9373a857a0
# ╠═b5b526dd-37ef-484c-97fd-2305f0d1d714
# ╠═059eceb1-39fd-4a5e-a787-23d2bbf9b547
# ╠═759eb280-f138-46f8-af4e-7447a38c4a9f
# ╠═424be202-5f13-46e1-a4db-8cbd6937fdcc
# ╠═ea07bf0c-552c-46fe-9431-4ca6a99e3647
# ╠═691593a3-4e68-4b3c-8020-ee07bf773486
# ╠═8947ed4c-ac37-4531-9252-63b195b73577
# ╠═7fca1678-2d50-4ade-be8b-43430f751fcf
# ╠═04a4723c-c9d3-46cd-8d22-48b9c992f366
# ╟─45be7b0b-c64d-4cfe-a399-de62e6b9094e
# ╠═a51bf9fd-11a6-48ba-9b0d-ab4397be015c
# ╠═c80b6d31-ba85-4d8a-bc28-20dde3f74985
# ╠═b3e4776d-8d03-47c2-8057-8b6e4f443b90
# ╟─424b9b91-9926-4c54-9f9d-329c974a8336
