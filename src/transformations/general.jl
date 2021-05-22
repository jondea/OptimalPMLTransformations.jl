
optimal_pml_transformation(field_fnc::AbstractFieldFunction, pml_geometry::PMLGeometry, ν) = optimal_pml_transformation(field_fnc, pml_geometry, ν)

function optimal_pml_transformation_solve(field_fnc::Function, ν_max::T,
    ν_vec::Union{Vector{T},Nothing}=nothing,
    tν_vec::Union{Vector{Complex{T}},Nothing}=nothing,
    ∂tν_∂ν_vec::Union{Vector{Complex{T}},Nothing}=nothing,
    ∂tν_∂ζ_vec::Union{Vector{Complex{T}},Nothing}=nothing;
    ν0=zero(T), tν0=zero(Complex{T}), field0=field_fnc(tν0), U_field=field_fnc(zero(Complex{T})), householder_order=3, ε=1.0e-12,
    show_trace=false, h_max=0.1, tν_jump_max=0.5, t_angle_jump_max=π/8, N_iter_max=10, silent_failure=false) where {T<:Real}

    # Last resort kill switch
    overall_iter = 0

    # Start with 0
    ν = ν0
    if !isnothing(ν_vec)
        resize!(ν_vec, 1)
        ν_vec[1] = ν
    end

    tν = tν0
    if !isnothing(tν_vec)
        resize!(tν_vec, 1)
        tν_vec[1] = ν
    end

    U=U_field.u

    field = field0
    field_prev = field0

    # Add to vector if provided
    if !isnothing(∂tν_∂ν_vec)
        resize!(∂tν_∂ν_vec, 1)
        ∂tν_∂ν_vec[1] = ∂tν_∂ν(field, U_field)
    end

    # Add to vector if provided
    if !isnothing(∂tν_∂ζ_vec)
        resize!(∂tν_∂ζ_vec, 1)
        ∂tν_∂ζ_vec[1] = ∂tν_∂ζ(field, U_field, ν)
    end

    f(field::NamedTuple)::Complex{T}    = field.u - U*(1-ν)
    df(field::NamedTuple)::Complex{T}   = field.∂u_∂tν
    ddf(field::NamedTuple)::Complex{T}  = field.∂2u_∂tν2
    dddf(field::NamedTuple)::Complex{T} = field.∂3u_∂tν3

    # Rescale f for error/objective
    normalised_f(field::NamedTuple) = abs(f(field)/U)

    # Tangent of tν
    t = ∂tν_∂ν(field_prev, U_field)
    t_prev = t

    tν_prev = tν
    ν_prev  = ν

    # Try to guess what will be a good size for the next step without going past our max
    h = min(h_max, 0.9*tν_jump_max/abs(t), ν_max-ν)

    # Keep going until we get to our target
    while ν < ν_max

        # Last resort kill switch
        overall_iter += 1
        if overall_iter > 1000
            if silent_failure
                break
            else
                error("Too many steps overall, quitting")
            end
        end

        # Step in ν
        ν = ν_prev + h

        # Predictor step in tν using tangent (Euler)
        tν = tν_prev + h*t

        # Evaluate field at new estimate
        field = field_fnc(tν)

        # Corrector iterations
        iter = 0
        while normalised_f(field) > ε
            dtν = -f(field)/df(field)

            if householder_order == 1
                tν = tν + dtν
            elseif householder_order == 2
                tν = tν + dtν*(1 + dtν*(ddf(field)/(2*df(field))))^(-1)
            elseif householder_order == 3
                tν = tν + dtν*(1 + dtν*(ddf(field)/(2*df(field)))) / (1 + (ddf(field)/df(field))*dtν + 1//6*(dddf(field)/df(field))*dtν^2)
            else
                error("Householder order $householder_order not implemented. Use 1, 2 or 3.")
            end
            # Calculate field and its derivatives at new point
            field = field_fnc(tν)

            # Keep track of number of iterations and exit to reduce stepsize if we do too many
            iter += 1
            if iter > N_iter_max
                break
            end
        end

        # Get tangent at new point
        t = ∂tν_∂ν(field, U_field)

        # Get angle between this and previous
        # Angle between a=re^(im θ) and b=ρe^(im φ) is angle(r/ρ e^(im (θ-φ)))
        t_angle_jump = abs(angle(t_prev/t))

        # Only accept this stepsize if the tangent and value hasn't changed too much
        if (   t_angle_jump <= t_angle_jump_max
            && abs(tν-tν_prev) <= tν_jump_max
            && normalised_f(field) <= ε )

            # Add values to vectors if provided
            if !isnothing(ν_vec     ) push!(ν_vec, ν) end
            if !isnothing(tν_vec    ) push!(tν_vec, tν) end
            if !isnothing(∂tν_∂ν_vec) push!(∂tν_∂ν_vec, ∂tν_∂ν(field,U_field)) end
            if !isnothing(∂tν_∂ζ_vec) push!(∂tν_∂ζ_vec, ∂tν_∂ζ(field,U_field,ν)) end

            # Reset stepsize, try to guess what will be a good size for the next step without going past our max
            h = min(h_max, 0.9*tν_jump_max/abs(t), ν_max-ν)

            # Store current as the starting point for next step
            field_prev = field
            t_prev  = t
            ν_prev  = ν
            tν_prev = tν
        else
            # Reject the step and reduce step size
            # Go back to previous
            ν  = ν_prev
            tν = tν_prev
            t = t_prev
            field = field_prev

            # Half the size of h
            h = h/2
        end

    end

    return tν, ∂tν_∂ν(field,U_field), ∂tν_∂ζ(field,U_field,ν), ν, field
end

function pole_newton_solve(field_fnc::Function, ν0::Real, ζ0::Real, tν0::Number; ε=1e-12)

    ν = ν0
    ζ = ζ0
    tν = tν0

    x = [ν, ζ, real(tν), imag(tν)]

    # Objectives, normalised by u
    f1(field::FieldAndDerivativesAtPoint) = field.∂u_∂tν
    f2(field::FieldAndDerivativesAtPoint, ν) = (field.u - U_field.u*(1-ν))

    U_field = field_fnc(0.0+0.0im, ζ)
    field = field_fnc(tν, ζ)

    # Create vector of residuals
    r = [real(f1(field)), imag(f1(field)), real(f2(field,ν)), imag(f2(field,ν))]

    counter = 1
    while maximum(abs.(r)) > ε*abs(U_field.u)

        # Create Jacobian of objectives and unknowns
        df2_dζ = field.∂u_∂tζ-U_field.∂u_∂tζ*(1-ν)
        J = [
            0               real(field.d2u_dtνdζ) real(field.∂2u_∂tν2) -imag(field.∂2u_∂tν2);
            0               imag(field.d2u_dtνdζ) imag(field.∂2u_∂tν2)  real(field.∂2u_∂tν2);
            real(U_field.u) real(df2_dζ)          real(field.∂u_∂tν)   -imag(field.∂u_∂tν)  ;
            imag(U_field.u) imag(df2_dζ)          imag(field.∂u_∂tν)    real(field.∂u_∂tν)
        ]

        # Perform Newton step
        x = x - J\r

        # Get values of unknowns from vector
        ν = x[1]
        ζ = x[2]
        tν = x[3] + im*x[4]

        # Recompute field and residual at new point
        U_field = field_fnc(0.0+0.0im, ζ)
        field = field_fnc(tν, ζ)
        r = [real(f1(field)), imag(f1(field)), real(f2(field,ν)), imag(f2(field,ν))]

        counter += 1
    end

    return ν, ζ, tν
end
