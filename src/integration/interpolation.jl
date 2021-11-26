
function integrate(intrp::InterpLine, f::Function; order=2)
    knots, weights = gausslegendre(order)
    knots .= (knots .+ 1)./2
    weights ./= 2
    # Note to self, write iterator over spline patches
    # so that this algorithm is O(N) rather than O(N^2)
    intrp_it = Base.Iterators.Stateful(intrp.points)
    intrp_prev = popfirst!(intrp_it)
    integrand = zero(f(intrp_prev))
    while !isempty(intrp_it)
        intrp_next = peek(intrp_it)
        h = intrp_next.ν - intrp_prev.ν
        for (knot, weight) in zip(knots, weights)
            ν = intrp_prev.ν + knot * h
            # TODO: Implement eval_hermite_patch which takes knot directly
            intrp_point = eval_hermite_patch(intrp_prev, intrp_next, ν)
            integrand += f(intrp_point)*weight*h
        end
        intrp_prev = popfirst!(intrp_it)
    end
    integrand
end

function integrate(tν_interp0::InterpLine, tν_interp1::InterpLine, f::Function; order=2)

    intrp_ν_both = vcat(tν_interp0.x, tν_interp1.x) |> sort |>
                    unique |> Base.Iterators.Stateful

    knots, weights = gausslegendre(order)
    # Note to self, write iterator over spline patches
    # so that this algorithm is O(N) rather than O(N^2)
    intrp_ν_prev = popfirst!(intrp_ν_both)
    integrand = zero(f(intrp_ν_prev))
    for intrp_ν_next in intrp_ν_both
        h = intrp_ν_next - intrp_ν_prev
        for (knot, weight) in zip(knots, weights)
            ν = intrp_ν_prev + knot * h/2
            integrand += f(tν_intrp0, tν_intrp, ν)*weight*h
        end
        intrp_ν_prev = intrp_ν_next
    end
    integrand
end

function integrate_between(tν_interp0::InterpLine, tν_interp1::InterpLine, f::Function; order=2)

    intrp_points0 = Base.Iterators.Stateful(tν_interp0.points)
    intrp_points1 = Base.Iterators.Stateful(tν_interp1.points)

    intrp_point0_prev = popfirst!(intrp_points0)
    intrp_point1_prev = popfirst!(intrp_points1)

    intrp00 = intrp_point0_prev
    intrp01 = intrp_point1_prev

    ν0 = min(intrp_point0_prev.ν, intrp_point1_prev.ν)

    ζ0 = tν_interp0.ζ
    ζ1 = tν_interp1.ζ

    # We could probably deal with this by extrapolation, but we shouldn't have
    @assert first(tν_interp0.points).ν == first(tν_interp1.points).ν
    @assert last(tν_interp0.points).ν == last(tν_interp1.points).ν

    intrp_point0_next = popfirst!(intrp_points0)
    intrp_point1_next = popfirst!(intrp_points1)

    intrp10 = intrp_point0_next
    intrp11 = intrp_point1_next
    ν1 = min(intrp01.ν, intrp11.ν)

    knots, weights = gausslegendre(order)
    knots .= (knots .+ 1)./2
    weights ./= 2

    function linear_interpolate_∂tν_∂ζ(p0::InterpPoint, p1::InterpPoint, ν::Number)
        t = (ν - p0.ν)/(p1.ν - p0.ν)
        (1-t)*p0.∂tν_∂ζ + t*p1.∂tν_∂ζ
    end

    integrand = zero(f(intrp00))

    last_patch = isempty(intrp_points0) && isempty(intrp_points1)
    done = false

    while !done
        if intrp_point0_next.ν > intrp_point1_next.ν
            intrp11 = intrp_point1_next
            ν1 = intrp11.ν
            # We don't have a point for 0, interpolate tν
            intrp10 = eval_hermite_patch(intrp_point0_next, intrp_point0_prev, ν1)

            intrp_point1_prev = intrp_point1_next
            intrp_point1_next = popfirst!(intrp_points1)

        elseif intrp_point1_next.ν > intrp_point0_next.ν
            intrp10 = intrp_point0_next
            ν1 = intrp10.ν
            # We don't have a point for 1, interpolate
            intrp11 = eval_hermite_patch(intrp_point1_next, intrp_point1_prev, ν1)

            intrp_point0_prev = intrp_point0_next
            intrp_point0_next = popfirst!(intrp_points0)
        else # Equal
            intrp10 = intrp_point0_next
            intrp11 = intrp_point1_next

            ν1 = intrp11.ν # == intrp10.ν

            # The end points must be equal on the last patch
            if !last_patch
                intrp_point0_next = popfirst!(intrp_points0)
                intrp_point1_next = popfirst!(intrp_points1)
            end
        end

        # Integrate over patch using hermite with intrp00 etc
        δν = ν1 - ν0
        δζ = ζ1 - ζ0

        if isinf(intrp10) || isinf(intrp11)
            # The transformation is unbounded at the outer edge so we have to extrapolate from the second to last point
            for (knot_ν, weight_ν) in zip(knots, weights)
                for (knot_ζ, weight_ζ) in zip(knots, weights)
                    ν = ν0 + knot_ν*δν
                    ζ = ζ0 + knot_ζ*δζ
                    intrp = InterpPoint(ν, cubic_linear_extrapolation(ν0, ζ0, ζ1, Dtν_νζ(intrp00), Dtν_νζ(intrp01), ν, ζ))
                    integrand += f(intrp) * weight_ν * weight_ζ * δν * δζ
                end
            end
        elseif isnan(intrp00.∂tν_∂ν) || isnan(intrp00.∂tν_∂ζ) ||
            isnan(intrp01.∂tν_∂ν) || isnan(intrp01.∂tν_∂ζ) ||
            isnan(intrp10.∂tν_∂ν) || isnan(intrp10.∂tν_∂ζ) ||
            isnan(intrp11.∂tν_∂ν) || isnan(intrp11.∂tν_∂ζ)
            # A derivative is undefined, happens around the root of a rip (a branch point)

            for (knot_ν, weight_ν) in zip(knots, weights)
                for (knot_ζ, weight_ζ) in zip(knots, weights)
                    ν = ν0 + knot_ν*δν
                    ζ = ζ0 + knot_ζ*δζ
                    intrp = InterpPoint(ν, bilinear_patch(ν0, ν1,  ζ0, ζ1, intrp00.tν, intrp10.tν, intrp01.tν, intrp11.tν, ν, ζ)...)
                    integrand += f(intrp) * weight_ν * weight_ζ * δν * δζ
                end
            end
        else
            for (knot_ν, weight_ν) in zip(knots, weights)
                for (knot_ζ, weight_ζ) in zip(knots, weights)
                    ν = ν0 + knot_ν*δν
                    ζ = ζ0 + knot_ζ*δζ
                    intrp = eval_hermite_patch(ζ0, ζ1, intrp00, intrp10, intrp01, intrp11, ν, ζ)
                    integrand += f(intrp) * weight_ν * weight_ζ * δν * δζ
                end
            end
        end
        done = last_patch
        last_patch = isempty(intrp_points0) && isempty(intrp_points1)

        # Prepare for next iteration
        ν0 = ν1
        intrp00 = intrp10
        intrp01 = intrp11
    end

    integrand
end

function integrate(region::ContinuousInterpolation, f::Function; order=2)
    lines = Base.Iterators.Stateful(region.lines)
    line_prev = popfirst!(lines)
    integrand = 0
    while !isempty(lines)
        line_next = peek(lines)
        integrand += integrate_between(line_prev, line_next, f; order)
        line_prev = popfirst!(lines)
    end
    integrand
end
