using Parameters

function setup_constraint_handler(dh::Ferrite.AbstractDofHandler, left_bc, right_bc)
	ch = ConstraintHandler(dh)

	gridfaceset(s) = getfaceset(dh.grid, s)

    # If the corresponding bc is not set, then it is unconstrained and therefore periodic
    include_min=isnothing(left_bc)
    include_max=isnothing(right_bc)
    add_periodic!(ch, [gridfaceset("bottom")=>gridfaceset("top")], x->x[1]; include_min, include_max)

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

                # ds1_dν = 2
                # ds2_dθ = 2/(θ_max - θ_min)
                # J_νs = diagm(Tensor{2,2}, [ds1_dν, ds2_dθ])

                u_pml = PMLFieldFunction(pml)

                # Used inside integrate_hcubature
                function contribution(tpoint::TransformationPoint)
                    ν = tpoint.ν
                    ζ = tpoint.ζ
                    (;r, θ) = convert(PolarCoordinates, PMLCoordinates(ν,ζ), pml.geom)

                    # If s ranges from -1 to 1, then conversion to ν should be linear 0-1 if we have 1 PML
                    # How do we know which way it is oriented?
                    s_θ = 2(θ - θ_min)/(θ_max - θ_min) - 1

                    # Create quad rule with a single point
                    point_quad_rule = QuadratureRule{1,RefCube,Float64}([1.0],[Tensors.Vec(s_θ)])
                    pml_cellvalues = FaceScalarValues(point_quad_rule, cellvalues.func_interp)
                    reinit!(pml_cellvalues, cell, face)

                    Ke = zeros(T, n_basefuncs, n_basefuncs)

                    tr, J_ = tr_and_jacobian(pml, tpoint)
                    J_pml = Tensors.Tensor{2,2,ComplexF64}(J_)
                    Jₜₓₜᵣ = diagm(Tensor{2,2}, [1.0, 1/tr])
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
                            Ke[i, j] += ((Jₜₓₜᵣ ⋅ Jₜᵣᵣ ⋅  ∇δu) ⋅ (Jₜₓₜᵣ ⋅ Jₜᵣᵣ ⋅ ∇u) - k^2*δu * u) * tr * detJₜᵣᵣ
                        end
                    end
                    return Ke
                end

                intrp = interpolation(PMLFieldFunction(pml), range(θ_min, θ_max, length=3))

                # If we have an actual PML element, we need to constrain dofs on
                # outer boundary I think that because Ferrite imposes BCs after
                # matrix setup, we can't use the usual way for these unbounded
                # integrals. To fix this, we have implemented it as a boundary
                # element and eliminated the dofs Could we just mask out the
                # values that correspond to the outer boundary? (Although this
                # doesn't fix the linear profile, that would require us to
                # implement a new element, which might throw off the
                # isoparametric stuff)

                # Initial estimate without correction to get a rough idea of magnitude
                Ke_est = integrate(intrp, contribution)
                rtol=1e-8
                atol=rtol*norm(Ke_est)

                # Ke += integrate_hcubature(intrp, contribution; atol=1e-12, rtol=1e-10, maxevals=1_000)
                Ke += integrate_hcubature(intrp, contribution; atol=rtol*norm(Ke_est), rtol, maxevals=1_000, corrector=u_pml)

                atol=rtol*norm(Ke)

                # Used inside integrate_hcubature
                function line_integrand(tpoint)
                    ν = tpoint.ν
                    ζ = tpoint.ζ
                    (;r, θ) = convert(PolarCoordinates, PMLCoordinates(ν,ζ), pml.geom)

                    # If s ranges from -1 to 1, then conversion to ν should be linear 0-1 if we have 1 PML
                    s_θ = 2(θ - θ_min)/(θ_max - θ_min) - 1

                    # Create quad rule with a single point
                    point_quad_rule = QuadratureRule{1,RefCube,Float64}([1.0],[Tensors.Vec(s_θ)])
                    pml_cellvalues = FaceScalarValues(point_quad_rule, cellvalues.func_interp)
                    reinit!(pml_cellvalues, cell, face)

                    Ke = zeros(T, n_basefuncs, n_basefuncs)

                    tr, J_ = tr_and_jacobian(pml, tpoint)
                    dtr_dν = J_[1,1]
                    J_pml = Tensors.Tensor{2,2,ComplexF64}(J_)
                    Jₜᵣᵣ = inv(J_pml)
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
                            # Second index is ∂/∂θ, which is the normal derivative
                            Ke[i, j] += (δu * ((Jₜᵣᵣ ⋅ ∇u)[2])) * dtr_dν / tr
                        end
                    end
                    return Ke
                end

                # Between each continuous region is a rip
                for (region1, region2) in consecutive_pairs(intrp.continuous_region)
                    tν₋ = last(region1.lines)
                    tν₊ = first(region2.lines)
                    Ke -= line_integrate_hcubature(tν₋, line_integrand; atol, rtol, maxevals=1_000, corrector=u_pml)
                    Ke += line_integrate_hcubature(tν₊, line_integrand; atol, rtol, maxevals=1_000, corrector=u_pml)
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

@with_kw mutable struct PMLHelmholtzPolarAnnulusParams{F,P,BQR,PQR}
    k::Float64
    N_θ::Int
    N_r::Int
    N_pml::Int
    cylinder_radius::Float64 = 1.0
    R::Float64 = 2.0
    δ_pml::Float64 = 1.0
    u_ana::F
    pml::P = nothing
    order::Int = 2
    bulk_qr::BQR
    pml_qr::PQR
end

@with_kw mutable struct Result
    assemble_time::Float64 = NaN
    solve_time::Float64 = NaN
    abs_sq_error::Float64 = NaN
    abs_sq_norm::Float64 = NaN
    rel_error::Float64 = NaN
end

# Very clever generic ways to print to CSV, but we actually need a custom function
csv_header(t) = csv_header(typeof(t))
csv_header(t::DataType) = join(string.(fieldnames(t)),',')

@generated function to_csv(p::P) where P
    ex = :(String[])
    for name in fieldnames(P)
        ex = :(vcat(string(p.$name), $ex))
    end
    return :(join($ex,','))
end

function solve(params::PMLHelmholtzPolarAnnulusParams; save_vtk=false)
    result = Result()
    dim=2

    @unpack k, N_θ, N_r, N_pml, cylinder_radius, R, δ_pml, u_ana, order, pml, bulk_qr, pml_qr = params

    ip = Lagrange{dim, RefCube, params.order}()
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

    cellvalues = CellScalarValues(bulk_qr, ip);
    pml_cellvalues = CellScalarValues(pml_qr, ip);
    K = create_sparsity_pattern(ch.dh, ch)
    result.assemble_time = @elapsed K, f = doassemble(cellvalues, pml_cellvalues, K, ch.dh, pml, k)
    apply!(K, f, ch)
    result.solve_time = @elapsed u = K \ f
    apply!(u, ch)
    result.abs_sq_error = integrate_solution((u,x)-> x∈pml ? 0.0 : abs(u - u_ana(x))^2, u, cellvalues, dh)
    result.abs_sq_norm = integrate_solution((u,x)-> x∈pml ? 0.0 : abs(u)^2, u, cellvalues, dh)
    result.rel_error = sqrt(result.abs_sq_error/result.abs_sq_norm)

    if save_vtk
        write_vtk("helmholtz", dh, u, u_ana)
    end

    return result
end

function solve(ch, cellvalues, pml_cellvalues)
	K = create_sparsity_pattern(ch.dh, ch)
	K, f = doassemble(cellvalues, pml_cellvalues, K, ch.dh)
    apply!(K, f, ch)
	u = K \ f
	apply!(u, ch)
	return u
end

