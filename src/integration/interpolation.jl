
# TODO:
# Refactor file using techniques from docs/examples/interpolation.jl
# Incorporate notebook into package now that we can "disable in file". Possibly split out the interpolation and integration parts

function integrate(intrp::InterpLine, f::Function; order=2)
    knots, weights = gausslegendreunit(order)

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
            # TODO: Implement hermite_interpolation which takes knot directly
            intrp_point = evaluate(InterpSegment(intrp_prev, intrp_next, intrp.ζ), ν)
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

function integrate_between(tν_interp0::InterpLine, tν_interp1::InterpLine, f::Function; corrector=nothing, order=2)

    ζ0 = tν_interp0.ζ
    ζ1 = tν_interp1.ζ

    # Get knots and weights and normalise for [0,1]
    knots, weights = gausslegendreunit(order)

    integral = zero(f(zero(TransformationPoint)))

    for patch in eachpatch(tν_interp0, tν_interp1)

        # Integrate over patch using hermite with intrp points
        ν0 = νmin(patch)
        ν1 = νmax(patch)
        δν = ν1 - ν0
        δζ = ζ1 - ζ0
        for (knot_ν, weight_ν) in zip(knots, weights)
            for (knot_ζ, weight_ζ) in zip(knots, weights)
                ν = ν0 + knot_ν*δν
                ζ = ζ0 + knot_ζ*δζ
                integral += f(patch(ν, ζ, corrector)) * weight_ν * weight_ζ * δν * δζ
            end
        end
    end

    integral
end

# Use consecutive pairs
function integrate(region::ContinuousInterpolation, f::Function; order=2)
    return mapreduce(+, consecutive_pairs(region.lines)) do (line_prev,line_next)
        integrate_between(line_prev, line_next, f; order)
    end
end

function integrate(intrp::Interpolation, f::Function; kwargs...)
    # Changed mapreduce to a for loop to make interactive debugging easier
    # mapreduce(region->integrate_hcubature(region, f; kwargs...), +, intrp.continuous_region)
    integral = integrate(first(intrp.continuous_region), f; kwargs...)
    for region in intrp.continuous_region[2:end]
        integral += integrate(region, f; kwargs...)
    end
    return integral
end


function integrate_between_hcubature(tν_interp0::InterpLine, tν_interp1::InterpLine, f::Function; corrector=nothing, kwargs...)

    integral = zero(f(zero(TransformationPoint)))

    for patch in eachpatch(tν_interp0, tν_interp1)
        integrand_fnc(νζ) = f(patch(νζ[1], νζ[2], corrector))
        integral += hcubature(integrand_fnc, SA[νmin(patch), ζmin(patch)], SA[νmax(patch), ζmax(patch)]; kwargs...)[1]
    end

    return integral
end

function integrate_hcubature(intrp::Interpolation, f::Function; kwargs...)
    # Changed mapreduce to a for loop to make interactive debugging easier
    # mapreduce(region->integrate_hcubature(region, f; kwargs...), +, intrp.continuous_region)
    integral = integrate_hcubature(first(intrp.continuous_region), f; kwargs...)
    for region in intrp.continuous_region[2:end]
        integral += integrate_hcubature(region, f; kwargs...)
    end
    return integral
end

# Change to use consecutive pairs, use firstrest
function integrate_hcubature(region::ContinuousInterpolation, f::Function; kwargs...)
    if length(region.lines) < 2
        error("Need at least 2 lines to integrate between them")
    end
    lines = Base.Iterators.Stateful(region.lines)
    line_prev = popfirst!(lines)
    line_next = peek(lines)
    integrand = integrate_between_hcubature(line_prev, line_next, f; kwargs...)
    line_prev = popfirst!(lines)
    while !isempty(lines)
        line_next = peek(lines)
        integrand += integrate_between_hcubature(line_prev, line_next, f; kwargs...)
        line_prev = popfirst!(lines)
    end
    return integrand
end

function line_integrate_hcubature(tν_interp0::InterpLine, tν_interp1::InterpLine, f::Function; kwargs...)

    # We could probably deal with this by extrapolation, but we shouldn't have to
    @assert first(tν_interp0.points).ν == first(tν_interp1.points).ν
    @assert last(tν_interp0.points).ν == last(tν_interp1.points).ν

    patches = eachpatch(tν_interp0, tν_interp1)
    patch, rest_of_patches = firstrest(patches)
    integral = hquadrature(ν->f(segment0(patch), segment1(patch),ν), νmin(patch), νmax(patch); kwargs...)[1]

    for patch in rest_of_patches
        if abs(measure(segment0(patch))) < 2*eps(patch.ν0) || abs(measure(segment1(patch))) < 2*eps(patch.ν0)
            continue
        end
        # Given that these patches are cubic at most, we could just use a known number of Gauss points
        # instead of an adaptive scheme. Of course we need an adaptive scheme if we use eval+correct
        integral += hquadrature(ν->f(segment0(patch), segment1(patch),ν), νmin(patch), νmax(patch); kwargs...)[1]
    end

    return integral
end


function line_integrate_hcubature(tν_interp::InterpLine, f::Function; corrector=nothing, kwargs...)

    ζ = tν_interp.ζ

    segments = consecutive_pairs(tν_interp.points)
    point_tuple, rest_of_segments = firstrest(segments)
    segment = InterpSegment(point_tuple[1], point_tuple[2], ζ)
    integral = hquadrature(ν->f(segment(ν, corrector)), νmin(segment), νmax(segment); kwargs...)[1]

    for point_tuple in rest_of_segments
        # Given that these patches are cubic at most, we could just use a known number of Gauss points
        # instead of an adaptive scheme. Of course we need an adaptive scheme if we use eval+correct
        segment = InterpSegment(point_tuple[1], point_tuple[2], ζ)
        integral += hquadrature(ν->f(segment(ν, corrector)), νmin(segment), νmax(segment); kwargs...)[1]
    end
    return integral
end
