

function interpolate(u::AbstractFieldFunction, pml::PMLGeometry, ζs, ν_max; δ = 1e-1, ε = 1e-4)

    function create_line(ζ)
        ν_vec = Float64[]
        tν_vec = ComplexF64[]
        ∂tν_∂ν_vec = ComplexF64[]
        ∂tν_∂ζ_vec = ComplexF64[]
        optimal_pml_transformation_solve(u, pml, ν_max, ζ, ν_vec, tν_vec, ∂tν_∂ν_vec, ∂tν_∂ζ_vec; silent_failure=true)
        # Add in point at ν=1? Try to work out if it's unbounded or not
        return InterpLine(ζ, ν_vec, tν_vec, ∂tν_∂ν_vec, ∂tν_∂ζ_vec)
    end

    function possible_rip_between(line1::InterpLine, line2::InterpLine)::Bool
        knots = 0:0.01:ν_max
        line1_points = line1.(knots)
        line2_points = line2.(knots)
        rel_diff = (2*norm(line1_points - line2_points)
            /(norm(line1_points) + norm(line2_points)))
        return rel_diff > δ
    end

    function recursively_subdivide(intrp, previous_line, next_line)
        ζ_mid = (previous_line.ζ + next_line.ζ)/2

        # We have to stop at some point and accept there's a rip
        if abs(previous_line.ζ - next_line.ζ) < ε
            p_rip₋ = argmax(p->abs(p.∂tν_∂ν)*(1-p.ν), previous_line.points)
            p_rip₊ = argmax(p->abs(p.∂tν_∂ν)*(1-p.ν), next_line.points)

            ν_rip = (p_rip₋.ν + p_rip₊.ν)/2
            tν_rip = (p_rip₋.tν + p_rip₊.tν)/2
            ζ_rip = ζ_mid

            ε_newton = 1e-12
            ν_rip, ζ_rip, tν_rip = pole_newton_solve(u, pml, ν_rip, ζ_mid, tν_rip; ε=ε_newton)

            rip_interp_point = InterpPoint(ν_rip, tν_rip, NaN+im*NaN, NaN+im*NaN)

            # Continue from the previous line to exactly on the rip, this should capture the boundary on
            # this side of the rip and will be the last line in the continuous region
            rip_line₋ = continue_in_ζ(u, pml, ζ_rip, previous_line)

            if possible_rip_between(previous_line, rip_line₋) error("ARGGGHH") end

            # Add the rip as a point, so that we can work around it later
            insertsorted!(rip_line₋.points, rip_interp_point; by=p->p.ν)

            push!(intrp, rip_line₋)

            # Add new rip, implicitly creating new continuous region
            push!(intrp, Rip(ζ_rip))

            # Continue from the next line to exactly on the rip, this should capture the boundary on
            # the other side of the rip, which will be the first line of the next continuous region
            rip_line₊ = continue_in_ζ(u, pml, ζ_rip, next_line)

            # Add the rip as a point, so that we can work around it later
            insertsorted!(rip_line₊.points, rip_interp_point; by=p->p.ν)

            push!(intrp, rip_line₊)

            return
        end

        mid_line = create_line(ζ_mid)

        if possible_rip_between(previous_line, mid_line)
            recursively_subdivide(intrp, previous_line, mid_line)
        end

        # When we get to this point, we have subdivided everything before the mid_line
        push!(intrp, mid_line)

        if possible_rip_between(mid_line, next_line)
            recursively_subdivide(intrp, mid_line, next_line)
        end
    end

    ζ_iter = Base.Iterators.Stateful(ζs)
    ζ = popfirst!(ζ_iter)
    previous_line = create_line(ζ)
    intrp = Interpolation(previous_line)

    for ζ in ζ_iter
        next_line = create_line(ζ)
        recursively_subdivide(intrp, previous_line, next_line)
        push!(intrp, next_line)
        previous_line = next_line
    end

    last(intrp.continuous_region).ζ₊ = last(ζs)

    return intrp
end


function eval_hermite_patch(ν0::Number, ν1::Number, p0::Dtν_ν, p1::Dtν_ν, ν::Number)

    tν0=p0.tν; ∂tν_∂ν0=p0.∂tν_∂ν;
    tν1=p1.tν; ∂tν_∂ν1=p1.∂tν_∂ν;

    δ = ν1 - ν0
    s = (ν - ν0) / δ
    h0, hd0, h1, hd1 = CubicHermiteSpline.basis(s)
    dh0, dhd0, dh1, dhd1 = CubicHermiteSpline.basis_derivative(s)

    tν     = tν0*h0  + δ*∂tν_∂ν0*hd0  + tν1*h1  + δ*∂tν_∂ν1*hd1
    dtν_ds = tν0*dh0 + δ*∂tν_∂ν0*dhd0 + tν1*dh1 + δ*∂tν_∂ν1*dhd1

    return tν, dtν_ds/δ
end

function eval_hermite_patch(p0::InterpPoint, p1::InterpPoint, ν::Number)

    ν0=p0.ν; tν0=p0.tν; ∂tν_∂ν0=p0.∂tν_∂ν; ∂tν_∂ζ0=p0.∂tν_∂ζ
    ν1=p1.ν; tν1=p1.tν; ∂tν_∂ν1=p1.∂tν_∂ν; ∂tν_∂ζ1=p1.∂tν_∂ζ

    δ = ν1 - ν0
    s = (ν - ν0) / δ
    if !isnan(∂tν_∂ν0) && !isnan(∂tν_∂ν1)
        h0, hd0, h1, hd1 = CubicHermiteSpline.basis(s)
        dh0, dhd0, dh1, dhd1 = CubicHermiteSpline.basis_derivative(s)

        tν     = tν0*h0  + δ*∂tν_∂ν0*hd0  + tν1*h1  + δ*∂tν_∂ν1*hd1
        dtν_ds = tν0*dh0 + δ*∂tν_∂ν0*dhd0 + tν1*dh1 + δ*∂tν_∂ν1*dhd1
    else
        tν     = tν0*(1-s)  + tν1*s
        dtν_ds = δ*∂tν_∂ν0*(1-s) + δ*∂tν_∂ν1*s
    end

    # Linear interpolation is best we can do with ζ
    ∂tν_∂ζ = (1-s)*∂tν_∂ζ0 + s*∂tν_∂ζ1

    return InterpPoint(ν, tν, dtν_ds/δ, ∂tν_∂ζ)
end

function InterpPoint(line::InterpLine, ν::Float64)::InterpPoint
    i = searchsortedfirst(line.points, (;ν=ν); by=t->t.ν)
    if i > length(line.points)
        throw(DomainError(ν, "is outside of sampled range, hence we cannot interpolate"))
    end
    p1 = line.points[i]
    if ν == p1.ν
        return p1
    end
    if i == 1
        throw(DomainError(ν, "is outside of sampled range, hence we cannot interpolate"))
    end
    p0 = line.points[i-1]

    return eval_hermite_patch(p0, p1, ν)
end

function (line::InterpLine)(ν::Float64)
    return InterpPoint(line, ν).tν
end

function ∂tν_∂ν(line::InterpLine, ν::Float64)
    return InterpPoint(line, ν).∂tν_∂ν
end


function eval_hermite_patch(ζ0::Number, ζ1::Number, p00::InterpPoint, p10::InterpPoint, p01::InterpPoint, p11::InterpPoint, ν::Number, ζ::Number)
    @assert p00.ν == p01.ν
    @assert p10.ν == p11.ν
    InterpPoint(ν, eval_hermite_patch(p00.ν, p10.ν, ζ0, ζ1, Dtν_νζ(p00), Dtν_νζ(p10), Dtν_νζ(p01), Dtν_νζ(p11), ν, ζ))
end

function eval_hermite_patch(ν0::Number, ν1::Number, ζ0::Number, ζ1::Number, p00::Dtν_νζ, p10::Dtν_νζ, p01::Dtν_νζ, p11::Dtν_νζ, ν::Number, ζ::Number)

    tν00=p00.tν; ∂tν_∂ν00=p00.∂tν_∂ν; ∂tν_∂ζ00=p00.∂tν_∂ζ
    tν10=p10.tν; ∂tν_∂ν10=p10.∂tν_∂ν; ∂tν_∂ζ10=p10.∂tν_∂ζ
    tν01=p01.tν; ∂tν_∂ν01=p01.∂tν_∂ν; ∂tν_∂ζ01=p01.∂tν_∂ζ
    tν11=p11.tν; ∂tν_∂ν11=p11.∂tν_∂ν; ∂tν_∂ζ11=p11.∂tν_∂ζ

    δν = ν1 - ν0
    sν = (ν - ν0) / δν
    h0ν, hd0ν, h1ν, hd1ν = CubicHermiteSpline.basis(sν)
    dh0ν, dhd0ν, dh1ν, dhd1ν = CubicHermiteSpline.basis_derivative(sν)

    δζ = ζ1 - ζ0
    sζ = (ζ - ζ0) / δζ
    h0ζ, hd0ζ, h1ζ, hd1ζ = CubicHermiteSpline.basis(sζ)
    dh0ζ, dhd0ζ, dh1ζ, dhd1ζ = CubicHermiteSpline.basis_derivative(sζ)

    tν = (tν00*h0ν*h0ζ + tν10*h1ν*h0ζ + tν01*h0ν*h1ζ + tν11*h1ν*h1ζ
          + δν*(∂tν_∂ν00*hd0ν*h0ζ  + ∂tν_∂ν10*hd1ν*h0ζ
				+ ∂tν_∂ν01*hd0ν*h1ζ  + ∂tν_∂ν11*hd1ν*h1ζ)
          + δζ*(∂tν_∂ζ00*h0ν*hd0ζ  + ∂tν_∂ζ10*h1ν*hd0ζ
				+ ∂tν_∂ζ01*h0ν*hd1ζ  + ∂tν_∂ζ11*h1ν*hd1ζ))
	dtν_dsν = (tν00*dh0ν*h0ζ + tν10*dh1ν*h0ζ + tν01*dh0ν*h1ζ + tν11*dh1ν*h1ζ
          + δν*(∂tν_∂ν00*dhd0ν*h0ζ  + ∂tν_∂ν10*dhd1ν*h0ζ
				+ ∂tν_∂ν01*dhd0ν*h1ζ  + ∂tν_∂ν11*dhd1ν*h1ζ)
          + δζ*(∂tν_∂ζ00*dh0ν*hd0ζ  + ∂tν_∂ζ10*dh1ν*hd0ζ
				+ ∂tν_∂ζ01*dh0ν*hd1ζ  + ∂tν_∂ζ11*dh1ν*hd1ζ))
	dtν_dsζ = (tν00*h0ν*dh0ζ + tν10*h1ν*dh0ζ + tν01*h0ν*dh1ζ + tν11*h1ν*dh1ζ
          + δν*(∂tν_∂ν00*hd0ν*dh0ζ  + ∂tν_∂ν10*hd1ν*dh0ζ
				+ ∂tν_∂ν01*hd0ν*dh1ζ  + ∂tν_∂ν11*hd1ν*dh1ζ)
          + δζ*(∂tν_∂ζ00*h0ν*dhd0ζ  + ∂tν_∂ζ10*h1ν*dhd0ζ
				+ ∂tν_∂ζ01*h0ν*dhd1ζ  + ∂tν_∂ζ11*h1ν*dhd1ζ))

    return Dtν_νζ(tν, dtν_dsν/δν, dtν_dsζ/δζ)
end

function bilinear_patch(ν0::Number, ν1::Number, ζ0::Number, ζ1::Number, p00::Number, p10::Number, p01::Number, p11::Number, ν::Number, ζ::Number)
    δν = ν1 - ν0
    sν0 = (ν - ν0) / δν
    sν1 = 1 - sν0
    dsν0 = 1 / δν
    dsν1 = -dsν0

    δζ = ζ1 - ζ0
    sζ0 = (ζ - ζ0) / δζ
    sζ1 = 1 - sζ0
    dsζ0 = 1 / δζ
    dsζ1 = -dsζ0

    tν     = p00*sν0*sζ0  + p10*sν1*sζ0  + p01*sν0*sζ1  + p11*sν1*sζ1
    ∂tν_∂ν = p00*dsν0*sζ0 + p10*dsν1*sζ0 + p01*dsν0*sζ1 + p11*dsν1*sζ1
    ∂tν_∂ζ = p00*sν0*dsζ0 + p10*sν1*dsζ0 + p01*sν0*dsζ1 + p11*sν1*dsζ1

    return tν, ∂tν_∂ν, ∂tν_∂ζ
end

"Cubic interpolation in one direction, linear extrapolation in another"
function cubic_linear_extrapolation(ν0::Number, ζ0::Number, ζ1::Number, p00::Dtν_νζ, p01::Dtν_νζ, ν::Number, ζ::Number)
    δν = ν1 - ν0
    sν0 = (ν - ν0) / δν
    hν = 1
    hdν = sν0

    δζ = ζ1 - ζ0
    sζ = (ζ - ζ0) / δζ

    h0ζ, hd0ζ, h1ζ, hd1ζ = CubicHermiteSpline.basis(sζ)
    dh0ζ, dhd0ζ, dh1ζ, dhd1ζ = CubicHermiteSpline.basis_derivative(sζ)

    tν = (p00.tν*hν*h0ζ +  p00.∂tν_∂ν*hdν*h0ζ + p00.∂tν_∂ζ*hν*hd0ζ
        + p01.tν*hν*h1ζ +  p01.∂tν_∂ν*hdν*h1ζ + p01.∂tν_∂ζ*hν*hd1ζ)

    ∂tν_∂ν = ( p00.∂tν_∂ν*h0ζ + p01.tν*h1ζ)

    ∂tν_∂sζ = (p00.tν*hν*dh0ζ +  p00.∂tν_∂ν*hdν*dh0ζ + p00.∂tν_∂ζ*hν*dhd0ζ
        + p01.tν*hν*dh1ζ +  p01.∂tν_∂ν*hdν*dh1ζ + p01.∂tν_∂ζ*hν*dhd1ζ)

    return Dtν_νζ(tν, ∂tν_∂ν, ∂tν_∂sζ/δζ)

end

function evaluate(patch::InterpPatch, ζ0, ζ1, ν, ζ)::InterpPoint
    intrp00 = patch.p00
    intrp01 = patch.p01
    intrp10 = patch.p10
    intrp11 = patch.p11

    ν0 = intrp00.ν
    ν1 = intrp10.ν

    if isinf(intrp10) || isinf(intrp11)
        # The transformation is unbounded at the outer edge so we have to extrapolate from the second to last point
        return InterpPoint(ν, cubic_linear_extrapolation(ν0, ζ0, ζ1, Dtν_νζ(intrp00), Dtν_νζ(intrp01), ν, ζ))
    elseif isnan(intrp00.∂tν_∂ν) || isnan(intrp00.∂tν_∂ζ) ||
           isnan(intrp01.∂tν_∂ν) || isnan(intrp01.∂tν_∂ζ) ||
           isnan(intrp10.∂tν_∂ν) || isnan(intrp10.∂tν_∂ζ) ||
           isnan(intrp11.∂tν_∂ν) || isnan(intrp11.∂tν_∂ζ)
        # A derivative is undefined, happens around the root of a rip (a branch point)
        return InterpPoint(ν, bilinear_patch(ν0, ν1,  ζ0, ζ1, intrp00.tν, intrp10.tν, intrp01.tν, intrp11.tν, ν, ζ)...)
    else
        return eval_hermite_patch(ζ0, ζ1, intrp00, intrp10, intrp01, intrp11, ν, ζ)
    end
end

function evalute_and_correct(u, pml, patch::InterpPatch, ζ0, ζ1, ν, ζ)
    intrp = evaluate(patch, ζ0, ζ1, ν, ζ)
    field_fnc(tν) = u(NamedTuple{(:u, :∂u_∂tν, :∂u_∂tζ, :∂2u_∂tν2, :∂2u_∂tν∂tζ, :∂3u_∂tν3)}, PMLCoordinates(tν, ζ), pml)
    U_field = field_fnc(zero(complex(ν)))
    tν, field, converged = corrector(field_fnc, U_field.u, ν, intrp.tν)
    if converged
        return InterpPoint(ν, tν, ∂tν_∂ν(field, U_field), ∂tν_∂ζ(field, U_field, ν))
    else
        return InterpPoint(ν, NaN+im*NaN, NaN+im*NaN, NaN+im*NaN)
    end
end
