
function optimal_pml_transformation_solve(u::AbstractFieldFunction, pml::PMLGeometry, ν_max, ζ, args...; kwargs...)
    u_pml_coords(tν) = u(NamedTuple{(:u, :∂u_∂tν, :∂u_∂tζ, :∂2u_∂tν2, :∂2u_∂tν∂tζ, :∂3u_∂tν3)}, PMLCoordinates(tν,ζ), pml)
    return optimal_pml_transformation_solve(u_pml_coords, ν_max, args...;kwargs...)
end

function corrector(field_fnc::Function, U::Number, ν, tν0, field0=field_fnc(tν0); N_iter_max=10, householder_order=3, ε=1e-12)

    f(field::NamedTuple)    = field.u - U*(1-ν)
    df(field::NamedTuple)   = field.∂u_∂tν
    ddf(field::NamedTuple)  = field.∂2u_∂tν2
    dddf(field::NamedTuple) = field.∂3u_∂tν3

    # Rescale f for error/objective
    normalised_f(field::NamedTuple) = abs(f(field)/U)

    field = field0
    tν = complex(tν0)

    # Corrector iterations
    iter = 0
    while normalised_f(field) > ε
        dtν = -f(field)/df(field)

        if isnan(dtν)
            error("dtν is nan, cannot continue")
        end
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
            return tν, field, false
        end
    end

    return tν, field, true
end

function optimal_pml_transformation_solve(field_fnc::Function, ν_max::T,
    ν_vec::Union{Vector{T},Nothing}=nothing,
    tν_vec::Union{Vector{Complex{T}},Nothing}=nothing,
    ∂tν_∂ν_vec::Union{Vector{Complex{T}},Nothing}=nothing,
    ∂tν_∂ζ_vec::Union{Vector{Complex{T}},Nothing}=nothing;
    ν0=zero(T), tν0=zero(Complex{T}), field0=field_fnc(tν0), U_field=field_fnc(zero(Complex{T})), householder_order=3, ε=1.0e-12,
    h_max=0.1, h_min=1e-12, tν_jump_max=10.0, tν_correction_ratio_max=0.2, t_angle_jump_max=π/8, N_iter_max=10) where {T<:Real}

    # Last resort kill switch
    overall_iter = 0

    if isnan(tν0)
        error("tν0 is nan, cannot continue")
    end

    if isnan(ν0)
        error("ν0 is nan, cannot continue")
    end

    function push_trace_vectors(ν, tν, field)
        if !isnothing(ν_vec     ) push!(ν_vec, ν) end
        if !isnothing(tν_vec    ) push!(tν_vec, tν) end
        if !isnothing(∂tν_∂ν_vec) push!(∂tν_∂ν_vec, ∂tν_∂ν(field,U_field)) end
        if !isnothing(∂tν_∂ζ_vec) push!(∂tν_∂ζ_vec, ∂tν_∂ζ(field,U_field,ν)) end
    end

    function ν_step(ν_prev, t, h_prev=Inf)::Tuple{T,T}
        # Could we do tν_jump_max as relative to abs(tν)?
        # Try to guess what will be a good size for the next step without going past our max
        h = min(h_max, 0.9*tν_jump_max/abs(t), 2*h_prev)
        ν = ν_prev + h
        # If the step will almost take us to the end, just go straight there.
        if (ν_max-ν) < (1 + sqrt(eps(ν_max)))*h_min
            h = ν_max-ν_prev
            ν = ν_max
        end
        return ν, h
    end

    ν::T = ν0

    tν = complex(tν0)

    U=U_field.u

    field = field0
    field_prev = field0

    push_trace_vectors(ν, tν, field)

    # Tangent of tν
    t = ∂tν_∂ν(field_prev, U_field)
    t_prev = t

    tν_prev = tν
    ν_prev  = ν
    ν, h = ν_step(ν, t)

    # Store h_prev for a kind of momentum, we half the failed step when things are bad,
    # but we at most double the previous when things look better
    h_prev = h

    # Keep going until we get to our target
    while ν_prev < ν_max

        if isnan(h)
            error("step (h) is nan, cannot continue stepping")
        end

        # Last resort break to stop infinite loop
        overall_iter += 1
        if overall_iter > 10000
            # This shouldn't happen, the step size checker should have caught this
            # throw a noisy error and fix it!
            error("Too many steps overall, quitting")
        end

        # Predictor step in tν using tangent (Euler)
        tν_pred = tν_prev + h*t
        tν = tν_pred

        # Evaluate field at new estimate
        field = field_fnc(tν)

        # Corrector iterations
        tν, field, converged = corrector(field_fnc, U, ν, tν, field; N_iter_max, householder_order, ε)

        # Get tangent at new point
        t = ∂tν_∂ν(field, U_field)

        # Get angle between this and previous
        # Angle between a=re^(im θ) and b=ρe^(im φ) is angle(r/ρ e^(im (θ-φ)))
        t_angle_jump = abs(angle(t_prev/t))

        # Only accept this stepsize if the tangent and value hasn't changed too much
        if (   t_angle_jump <= t_angle_jump_max
            && abs(tν-tν_prev) <= tν_jump_max
            && abs(tν-tν_pred)/abs(tν_pred - tν_prev) <= tν_correction_ratio_max
            && converged )

            push_trace_vectors(ν, tν, field)

            if ν == ν_max
                break
            end

            # Store current as the starting point for next step
            field_prev = field
            t_prev  = t
            ν_prev  = ν
            tν_prev = tν
            h_prev = h

            ν, h = ν_step(ν, t, h_prev)

        else

            @debug "Rejecting at ν=$ν, h=$h, overall_iter=$overall_iter reasons: "
            if !(t_angle_jump <= t_angle_jump_max)
                @debug "    t_angle_jump too big: $t_angle_jump "
            end
            if !(abs(tν-tν_prev) <= tν_jump_max)
                @debug "    abs(tν-tν_prev) too big: $(abs(tν-tν_prev)) "
            end
            if !(converged)
                @debug "    not converged "
            end

            # Reject the step and reduce step size
            # Go back to previous
            ν  = ν_prev
            tν = tν_prev
            t = t_prev
            field = field_prev

            # Half the size of h
            h = h/2
            ν = ν + h
        end

        # Stepsize is too small, give up and mark end as NaN
        # This is expected if the transformation is unbounded at the outer edge
        # so we don't throw
        if h < h_min
            ν = ν_max
            tν = T(NaN) + im*T(NaN)
            field = field_fnc(tν)
            push_trace_vectors(ν, tν, field)
            break
        end

    end

    return tν, ∂tν_∂ν(field,U_field), ∂tν_∂ζ(field,U_field,ν), ν, field
end


"Continue from p0 to ζ"
function continue_in_ζ(field_fnc::Function, ζ::Number, p0::InterpPoint;
    field=field_fnc(p0.tν, ζ), U_field=field_fnc(zero(p0.tν),ζ),
    householder_order=3, N_iter_max=10, ε=1e-12)

    ν = p0.ν
    tν0 = p0.tν
    U = U_field.u
    tν, field, converged = corrector(tν->field_fnc(tν,ζ), U, ν, tν0, field; N_iter_max, householder_order, ε)
    if !converged error("Did not converge") end
    InterpPoint(ν, tν, ∂tν_∂ν(field, U_field), ∂tν_∂ζ(field, U_field,ν))
end

function continue_in_ζ(u::AbstractFieldFunction, pml::PMLGeometry, args...; kwargs...)
    u_pml_coords(tν,ζ) = u(NamedTuple{(:u, :∂u_∂tν, :∂u_∂tζ, :∂2u_∂tν2, :∂2u_∂tν∂tζ, :∂3u_∂tν3)}, PMLCoordinates(tν,ζ), pml)
    return continue_in_ζ(u_pml_coords, args...; kwargs...)
end

"Continue from each point in tν0 to ζ"
function continue_in_ζ(field_fnc::Function, ζ::Number, tν0::InterpLine; householder_order=3, N_iter_max=10, ε=1e-12)
    U_field = field_fnc(zero(first(tν0.points).tν), ζ)

    return InterpLine(
        ζ,
        [continue_in_ζ(field_fnc, ζ, p; U_field, N_iter_max, householder_order, ε) for p in tν0.points]
    )
end

continue_in_ζ(u_pml::PMLFieldFunction, args...; kwargs...) = continue_in_ζ(u_pml.u, u_pml.pml, args...; kwargs...)
