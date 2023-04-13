

"Adaptively append interpolation lines to the interpolation, between first(ζs) and last(ζs) (not including end points)"
function adaptively_append!(intrp::Interpolation, u_pml::PMLFieldFunction, ζs::AbstractVector;
	ε=1e-1, δ=1e-8, tν_metric=relative_l2_difference,
	tν₊=interpolation(u_pml, last(ζs)))::Interpolation

	tν₁ = last(last(intrp.continuous_region).lines)
	for (ζ₁, ζ₂) in consecutive_pairs(ζs)
		tν₂ = (ζ₂ == last(ζs)) ? tν₊ : interpolation(u_pml, ζ₂)
		if tν_metric(tν₁,tν₂) > ε
			if abs(ζ₂ - ζ₁) < δ
				# Find rip point accurately using Newton's method
				rip = find_rip(u_pml, tν₁, tν₂)

				# Rip has unbounded derivatives in ν and ζ, add NaNs
				rip_interp_point = InterpPoint(rip.ν, rip.tν, NaN+im*NaN, NaN+im*NaN)

				# Add line on rip, continued from below
				tν_rip₋ = continue_in_ζ(u_pml, rip.ζ, tν₁)
				insertsorted!(tν_rip₋.points, rip_interp_point; by=p->p.ν)
				push!(intrp, tν_rip₋)

				# Adding a rip severs the two continuous regions
				push!(intrp, Rip(rip.ζ))

				# Add line on rip, continued from above
				tν_rip₊ = continue_in_ζ(u_pml, rip.ζ, tν₂)
				insertsorted!(tν_rip₊.points, rip_interp_point; by=p->p.ν)
				push!(intrp, tν_rip₊)
			else
				# Recursively split domain into 2 regions (3 points with ends)
				adaptively_append!(intrp, u_pml, range(ζ₁, ζ₂, length=3); ε, δ, tν₊=tν₂)
			end
		end
		# Don't add interpolation on the last ζ, or we'd add duplicates all up the call stack
		if ζ₂ != last(ζs)
			push!(intrp, tν₂)
		end
		tν₁ = tν₂
	end

	return intrp
end

"""
Find interpolation of `u_pml` at all ζs, adapting in ν and ζ

Subdivide in ζ if the tν_metric between two consecutive lines is >ε.
Add a rip if the division in ζ is <δ.
"""
function interpolation(u_pml::PMLFieldFunction, ζs::AbstractVector; kwargs...)::Interpolation

	# Start off interpolation with the first ζ
	tν₋ = interpolation(u_pml, first(ζs))
	intrp = Interpolation(tν₋)

	# Compute the interpolation at the last point and pass in for efficiency
	tν₊ = interpolation(u_pml, last(ζs))
	adaptively_append!(intrp, u_pml, ζs; tν₊, kwargs...)

	# Now we've done adaptively appending to the interpolation, add the final interpolation line
	push!(intrp, tν₊)

	return intrp

end

function interpolation(u_pml::PMLFieldFunction, ζ::Number; ν_max=1.0, kwargs...)
	νs = Float64[]
	tνs = ComplexF64[]
	∂tν_∂νs = ComplexF64[]
	∂tν_∂ζs = ComplexF64[]
	optimal_pml_transformation_solve(u_pml.u, u_pml.pml, ν_max, ζ, νs, tνs, ∂tν_∂νs, ∂tν_∂ζs; kwargs...)
	return InterpLine(ζ, νs, tνs, ∂tν_∂νs, ∂tν_∂ζs)
end

function hermite_interpolation(ν0::Number, ν1::Number, p0::Dtν_ν, p1::Dtν_ν, ν::Number)

    tν0=p0.tν; ∂tν_∂ν0=p0.∂tν_∂ν;
    tν1=p1.tν; ∂tν_∂ν1=p1.∂tν_∂ν;

    δ = ν1 - ν0
    s = (ν - ν0) / δ
    h0, hd0, h1, hd1 = CubicHermiteSpline.basis(s)
    dh0, dhd0, dh1, dhd1 = CubicHermiteSpline.basis_derivative(s)

    tν     = tν0*h0  + δ*∂tν_∂ν0*hd0  + tν1*h1  + δ*∂tν_∂ν1*hd1
    dtν_ds = tν0*dh0 + δ*∂tν_∂ν0*dhd0 + tν1*dh1 + δ*∂tν_∂ν1*dhd1

    return Dtν_ν(tν, dtν_ds/δ)
end

linear_interpolation(x0, x1, f0, f1, x) = f1*(x-x0)/(x1-x0) + f0*(x1-x)/(x1-x0)

function linear_interpolation(p0::InterpPoint, p1::InterpPoint, ν::Number)
    return InterpPoint(
        ν,
        linear_interpolation(p0.ν, p1.ν, p0.tν,     p1.tν,     ν),
        linear_interpolation(p0.ν, p1.ν, p0.∂tν_∂ν, p1.∂tν_∂ν, ν),
        linear_interpolation(p0.ν, p1.ν, p0.∂tν_∂ζ, p1.∂tν_∂ζ, ν)
    )
end

function robust_linear_interpolation(x0, x1, f0, f1, x)
    if     isnan(f0) return f1
    elseif isnan(f1) return f0
    else             return linear_interpolation(x0, x1, f0, f1, x)
    end
end

function robust_linear_interpolation(p0::InterpPoint, p1::InterpPoint, ν::Number)
    return InterpPoint(
        ν,
        robust_linear_interpolation(p0.ν, p1.ν, p0.tν,     p1.tν,     ν),
        robust_linear_interpolation(p0.ν, p1.ν, p0.∂tν_∂ν, p1.∂tν_∂ν, ν),
        robust_linear_interpolation(p0.ν, p1.ν, p0.∂tν_∂ζ, p1.∂tν_∂ζ, ν)
    )
end

# First order Taylor series
linear_extrapolation(x0, f0, df_dx0, x) = f0 + df_dx0*(x-x0)

# This handles nans, but the other evaluates don't, probably should be consistent here
function robust_hermite_interpolation(p0::InterpPoint, p1::InterpPoint, ν::Number)

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

    return robust_hermite_interpolation(p0, p1, ν)
end

function (line::InterpLine)(ν::Float64)
    return InterpPoint(line, ν).tν
end

function ∂tν_∂ν(line::InterpLine, ν::Float64)
    return InterpPoint(line, ν).∂tν_∂ν
end

function hermite_interpolation(patch::InterpPatch, ν::Number, ζ::Number)
    return hermite_interpolation(patch.ν0, patch.ν1, patch.ζ0, patch.ζ1, patch.p00, patch.p10, patch.p01, patch.p11, ν, ζ)
end

function hermite_interpolation(ν0::Number, ν1::Number, ζ0::Number, ζ1::Number, p00::Dtν_νζ, p10::Dtν_νζ, p01::Dtν_νζ, p11::Dtν_νζ, ν::Number, ζ::Number)

    tν00=p00.tν; ∂tν_∂ν00=p00.∂tν_∂ν; ∂tν_∂ζ00=p00.∂tν_∂ζ
    tν10=p10.tν; ∂tν_∂ν10=p10.∂tν_∂ν; ∂tν_∂ζ10=p10.∂tν_∂ζ
    tν01=p01.tν; ∂tν_∂ν01=p01.∂tν_∂ν; ∂tν_∂ζ01=p01.∂tν_∂ζ
    tν11=p11.tν; ∂tν_∂ν11=p11.∂tν_∂ν; ∂tν_∂ζ11=p11.∂tν_∂ζ

    δν = ν1 - ν0
    δν = δν == 0 ? copysign(eps(δν),δν) : δν # Stop NaNs for edge case of ν1 == ν0
    sν = (ν - ν0) / δν
    h0ν, hd0ν, h1ν, hd1ν = CubicHermiteSpline.basis(sν)
    dh0ν, dhd0ν, dh1ν, dhd1ν = CubicHermiteSpline.basis_derivative(sν)

    δζ = ζ1 - ζ0
    δζ = δζ == 0 ? copysign(eps(δζ),δζ) : δζ # Stop NaNs for edge case of ζ1 == ζ0
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
    sν0 = ν - ν0
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

function evaluate(patch::InterpPatch, ν, ζ)::TransformationPoint
    intrp00 = patch.p00
    intrp01 = patch.p01
    intrp10 = patch.p10
    intrp11 = patch.p11

    ζ0 = patch.ζ0
    ζ1 = patch.ζ1

    ν0 = patch.ν0
    ν1 = patch.ν1

    if !isfinite(intrp10.tν) || !isfinite(intrp11.tν)
        # The transformation is unbounded at the outer edge so we have to extrapolate from the second to last point
        return TransformationPoint(ν, ζ, cubic_linear_extrapolation(ν0, ζ0, ζ1, intrp00, intrp01, ν, ζ))
    elseif isnan(intrp00.∂tν_∂ν) || isnan(intrp00.∂tν_∂ζ) ||
           isnan(intrp01.∂tν_∂ν) || isnan(intrp01.∂tν_∂ζ) ||
           isnan(intrp10.∂tν_∂ν) || isnan(intrp10.∂tν_∂ζ) ||
           isnan(intrp11.∂tν_∂ν) || isnan(intrp11.∂tν_∂ζ)
        # A derivative is undefined, happens around the root of a rip (a branch point)
        return TransformationPoint(ν, ζ, bilinear_patch(ν0, ν1,  ζ0, ζ1, intrp00.tν, intrp10.tν, intrp01.tν, intrp11.tν, ν, ζ)...)
    else
        return TransformationPoint(ν, ζ, hermite_interpolation(patch, ν, ζ))
    end
end

(patch::InterpPatch)(ν::Number, ζ::Number) = evaluate(patch, ν, ζ)
(patch::InterpPatch)(ν::Number, ζ::Number, ::Nothing) = evaluate(patch, ν, ζ)
(patch::InterpPatch)(ν::Number, ζ::Number, corrector) = evaluate_and_correct(patch, ν, ζ, corrector)

function correct(intrp::TransformationPoint, u_pml::PMLFieldFunction)
    ν = intrp.ν; ζ = intrp.ζ
    field_fnc(tν) = u_pml(NamedTuple{(:u, :∂u_∂tν, :∂u_∂tζ, :∂2u_∂tν2, :∂2u_∂tν∂tζ, :∂3u_∂tν3)}, tν, ζ)
    U_field = field_fnc(zero(complex(ν)))
    tν, field, converged = corrector(field_fnc, U_field.u, ν, intrp.tν)
    return TransformationPoint(ν, ζ, tν, ∂tν_∂ν(field, U_field), ∂tν_∂ζ(field, U_field, ν)), converged
end

function evaluate_and_correct(patch::InterpPatch, ν, ζ, u_pml::PMLFieldFunction)
    intrp = evaluate(patch, ν, ζ)
    if !isfinite(patch)
        return intrp
    end
    if !isfinite(intrp)
        @warn "Patch at $ν and $ζ evaluated to non-finite value, there's no hope for the corrector"
        return intrp
    end
    corrected_intrp, converged = correct(intrp, u_pml)
    if converged && isfinite(corrected_intrp)
        return corrected_intrp
    else
        @warn "Corrector at $ν and $ζ did not converge for patch, using interpolated"
        return intrp
    end
end


function evaluate(segment::InterpSegment, ν)::TransformationPoint

    p0 = segment.p0
    p1 = segment.p1

    ν0 = p0.ν
    ν1 = p1.ν

    ζ = segment.ζ

    # special cases we want to handle differently
    if !isfinite(p1.tν)
        # p1 is not finite, extrapolate from p0 (this happens on the outer edge)
        tν = linear_extrapolation(ν0, p0.tν, p0.∂tν_∂ν, ν)
        return TransformationPoint(ν, ζ, tν, p0.∂tν_∂ν, p0.∂tν_∂ζ)
    elseif !isfinite(p0.∂tν_∂ν) || !isfinite(p0.∂tν_∂ζ) ||
           !isfinite(p1.∂tν_∂ν) || !isfinite(p1.∂tν_∂ζ)
        # Some derivatives are bad, fall back to linear interpolation
        return TransformationPoint(robust_linear_interpolation(p0, p1, ν), ζ)
    else
        # Everything else, note that there may still be NaNs/Infs, and they just propagate
        (;tν, ∂tν_∂ν) = hermite_interpolation(ν0, ν1, Dtν_ν(p0), Dtν_ν(p1), ν)
        ∂tν_∂ζ = robust_linear_interpolation(ν0, ν1, p0.∂tν_∂ζ, p1.∂tν_∂ζ, ν)
        return TransformationPoint(ν, ζ, tν, ∂tν_∂ν, ∂tν_∂ζ)
    end
end

(segment::InterpSegment)(ν::Number) = evaluate(segment, ν)
(segment::InterpSegment)(ν::Number, ::Nothing) = evaluate(segment, ν)
(segment::InterpSegment)(ν::Number, corrector) = evaluate_and_correct(segment, ν, corrector)

function evaluate_and_correct(segment::InterpSegment, ν, u_pml::PMLFieldFunction)
    intrp = evaluate(segment, ν)
    if !isfinite(segment)
        return intrp
    end
    if !isfinite(intrp)
        @warn "Segment at $ν and $ζ evaluated to non-finite value, there's no hope for the corrector"
        return intrp
    end
    corrected_intrp, converged = correct(intrp, u_pml)
    if converged && isfinite(corrected_intrp)
        return corrected_intrp
    else
        @warn "Corrector at $ν and $ζ did not converge for segment, using interpolated"
        return intrp
    end
end
