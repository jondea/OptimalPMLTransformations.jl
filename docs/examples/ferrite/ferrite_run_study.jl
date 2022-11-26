
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

function setup_constraint_handler(dh::Ferrite.AbstractDofHandler, left_bc, right_bc)
	ch = ConstraintHandler(dh)

	gridfaceset(s) = getfaceset(dh.grid, s)

    add_periodic!(ch, [gridfaceset("bottom")=>gridfaceset("top")], x->x[1])

    if !isnothing(left_bc)
        inner_dbc = Dirichlet(:u, gridfaceset("left"), left_bc)
        add!(ch, inner_dbc)
    end

    if !isnothing(right_bc)
        outer_dbc = Dirichlet(:u, gridfaceset("right"), right_bc)
        add!(ch, outer_dbc)
    end

    close!(ch)
	Ferrite.update!(ch, 0.0)
	ch
end

(f::AbstractFieldFunction)(c::Node) = f(c.x)

# Ignore time dependence (problem is time harmonic)
(f::AbstractFieldFunction)(x::Vec{2}, _t::Number) = f(x)

# Interpret Vec (the problem coordinates) as polar coordinates for the field function
(f::AbstractFieldFunction)(x::Vec{2}) = f(PolarCoordinates(x[1],x[2]))

Base.in(n::Node, pml::PMLGeometry) = in(n.x, pml)

Base.in(x::AbstractVector, pml::AnnularPML) = x[1] >= pml.R

Base.in(x::AbstractVector, pml::SFB) = in(x, pml.geom)

Base.in(x::AbstractVector, pml::PML) = in(x, pml.geom)

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

function doassemble(cellvalues::CellScalarValues{dim}, ::CellScalarValues{dim}, K::SparseMatrixCSC, dh::DofHandler, pml::InvHankelSeriesPML, k::Number) where {dim}
    doassemble(cellvalues, K, dh, pml, k)
end

consecutive_pairs(r) = IterTools.partition(r, 2, 1)

function doassemble(cellvalues::CellScalarValues{dim}, K::SparseMatrixCSC, dh::DofHandler, pml::InvHankelSeriesPML, k::Number) where {dim}

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

        # Collapse the PML into a boundary element. We can do this because a linear profile + pinned outer boundary
        # means there are no extra degrees of freedom.
        for face in 1:nfaces(cell)
            if onboundary(cell, face) && ((cellcount, face) ∈ getfaceset(dh.grid, "right"))

                # Get range of θ this element covers, this should create a tight covering because
                # nodes are shared between elements, so no rips should get through the floating point cracks
                θ_min, θ_max = extrema(c->c[2], coords)

                # Used inside integrate_hcubature
                function contribution(patch, ν, ζ)
                    (;r, θ) = convert(PolarCoordinates, PMLCoordinates(ν,ζ), pml.geom)

                    # If s ranges from -1 to 1, then conversion to ν should be linear 0-1 if we have 1 PML
                    s_θ = 2(θ - θ_min)/(θ_max - θ_min) - 1

                    # Create quad rule with a single point
                    point_quad_rule = QuadratureRule{1,RefCube,Float64}([1.0],[Tensors.Vec(s_θ)])
                    pml_cellvalues = FaceScalarValues(point_quad_rule, cellvalues.func_interp)
                    reinit!(pml_cellvalues, cell, face)

                    Ke = zeros(T, n_basefuncs, n_basefuncs)

                    tr, J_ = tr_and_jacobian(pml, patch, PolarCoordinates(r, θ))
                    J_pml = Tensors.Tensor{2,2,ComplexF64}(J_)
                    Jᵣₓ = diagm(Tensor{2,2}, [1.0, 1/tr])
                    Jₜᵣᵣ = inv(J_pml)
                    detJₜᵣᵣ	= det(J_pml)
                    for i in 1:n_basefuncs
                        # Shape function is defined on face, we use ν to extend into the collapsed PML
                        δU = shape_value(pml_cellvalues, 1, i)
                        ∇δU = shape_gradient(pml_cellvalues, 1, i)
                        δu = δU*(1-ν)
                        ∇δu = Tensors.Vec(-δU, ∇δU[2]*(1-ν))

                        for j in 1:n_basefuncs
                            # Shape function is defined on face, we use ν to extend into the collapsed PML
                            U = shape_value(pml_cellvalues, 1, j)
                            ∇U = shape_gradient(pml_cellvalues, 1, j)
                            u = U*(1-ν)
                            ∇u = Tensors.Vec(-U, ∇U[2]*(1-ν))

                            # Note there is no dΩ scaling because this is handled in the integrate function which calls this
                            Ke[i, j] += ((Jₜᵣᵣ ⋅ Jᵣₓ ⋅ ∇δu) ⋅ (Jₜᵣᵣ ⋅ Jᵣₓ⋅∇u) - k^2*δu * u) * tr * detJₜᵣᵣ
                        end
                    end
                    return Ke
                end

                ν_max = 1.0
                intrp = interpolate(pml.u, pml.geom, range(θ_min, θ_max, length=3), ν_max; h_min=1e-8)

                # If we have an actual PML element, we need to constrain dofs on
                # outer boundary I think that because Ferrite imposes BCs after
                # matrix setup, we can't use the usual way for these unbounded
                # integrals. To fix this, we have implemented it as a boundary
                # element and eliminated the dofs Could we just mask out the
                # values that correspond to the outer boundary? (Although this
                # doesn't fix the linear profile, that would require us to
                # implement a new element, which might throw off the
                # isoparametric stuff)
                Ke += integrate_hcubature(intrp, contribution; atol=1e-12, rtol=1e-10, maxevals=10_000)

                # Used inside integrate_hcubature
                function jump_contribution(segment, ν)
                    ζ = segment.ζ
                    (;r, θ) = convert(PolarCoordinates, PMLCoordinates(ν,ζ), pml.geom)

                    # If s ranges from -1 to 1, then conversion to ν should be linear 0-1 if we have 1 PML
                    s_θ = 2(θ - θ_min)/(θ_max - θ_min) - 1

                    # Create quad rule with a single point
                    point_quad_rule = QuadratureRule{1,RefCube,Float64}([1.0],[Tensors.Vec(s_θ)])
                    pml_cellvalues = FaceScalarValues(point_quad_rule, cellvalues.func_interp)
                    reinit!(pml_cellvalues, cell, face)

                    Ke = zeros(T, n_basefuncs, n_basefuncs)

                    tr, J_ = tr_and_jacobian(pml, segment, PolarCoordinates(r, θ))
                    dtr_dν = J_[1,1]
                    J_pml = Tensors.Tensor{2,2,ComplexF64}(J_)
                    Jᵣₓ = diagm(Tensor{2,2}, [1.0, 1/tr])
                    Jₜᵣᵣ = inv(J_pml)
                    dtx_dtr = Tensors.Vec{2}(-sin(θ), cos(θ))
                    for i in 1:n_basefuncs
                        # Shape function is defined on face, we use ν to extend into the collapsed PML
                        δU = shape_value(pml_cellvalues, 1, i)
                        ∇δU = shape_gradient(pml_cellvalues, 1, i)
                        δu = δU*(1-ν)
                        ∇δu = Tensors.Vec(-δU, ∇δU[2]*(1-ν))

                        for j in 1:n_basefuncs
                            # Shape function is defined on face, we use ν to extend into the collapsed PML
                            U = shape_value(pml_cellvalues, 1, j)
                            ∇U = shape_gradient(pml_cellvalues, 1, j)
                            u = U*(1-ν)
                            ∇u = Tensors.Vec(-U, ∇U[2]*(1-ν))

                            Ke[i, j] += (δu * ((Jₜᵣᵣ ⋅ Jᵣₓ⋅∇u) ⋅dtx_dtr)) *dtr_dν
                        end
                    end
                    return Ke
                end

                # Between each continuous region is a rip
                for (region1, region2) in consecutive_pairs(intrp.continuous_region)
                    tν₋ = last(region1.lines)
                    tν₊ = first(region2.lines)
                    Ke -= line_integrate_hcubature(tν₋, tν₊, jump_contribution; atol=1e-12, rtol=1e-10, maxevals=10_000)
                end
            end
        end

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

        celldofs!(global_dofs, cell)
        assemble!(assembler, global_dofs, fe, Ke)
    end
    return K, f
end

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
        # If N_pml == 0, then PML will be applied as a boundary condition
        if N_pml == 0
            ch = setup_constraint_handler(dh, u_ana, nothing)
        else
		    ch = setup_constraint_handler(dh, u_ana, (x,t)->zero(ComplexF64))
        end

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
    return (;assemble_time, solve_time, abs_sq_error, abs_sq_norm, rel_error)
end

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

function solve(ch, cellvalues, pml_cellvalues)
	K = create_sparsity_pattern(ch.dh, ch)
	K, f = doassemble(cellvalues, pml_cellvalues, K, ch.dh)
    apply!(K, f, ch)
	u = K \ f
	apply!(u, ch)
	return u
end
