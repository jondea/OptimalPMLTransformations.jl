

function integrate(tν_interp::InterpLine, f::Function; order=2)
    knots, weights = gausslegendre(order)
    # Note to self, write iterator over spline patches
    # so that this algorithm is O(N) rather than O(N^2)
    intrp_ν = Base.Iterators.Stateful(tν_interp.x)
    intrp_ν_prev = popfirst!(intrp_ν)
    integrand = zero(f(intrp, intrp_ν_prev))
    for intrp_ν_next in intrp_ν
        h = intrp_ν_next - intrp_ν_prev
        for (knot, weight) in zip(knots, weights)
            ν = intrp_ν_prev + knot * h/2
            integrand += f(tν_intrp(ν))*weight*h
        end
        intrp_ν_prev = intrp_ν_next
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
                else
            end
        end

        # Integrate over patch using hermite with intrp00 etc
        δν = ν1 - ν0
        δζ = ζ1 - ζ0

        for (knot_ν, weight_ν) in zip(knots, weights)
            for (knot_ζ, weight_ζ) in zip(knots, weights)
                ν = ν0 + knot_ν*δν
                ζ = ζ0 + knot_ζ*δζ
                intrp = eval_hermite_patch(ζ0, ζ1, intrp00, intrp10, intrp01, intrp11, ν, ζ)
                integrand += f(intrp) * weight_ν * weight_ζ * δν * δζ
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
