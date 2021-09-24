struct Rip
    ζ::Float64
end

struct InterpolationThroughPML
    ζ::Float64
    tν::CubicHermiteSplineInterpolation
end

(line::InterpolationThroughPML)(ν::Float64) = line.tν(ν)
∂tν_∂ν(line::InterpolationThroughPML, ν::Float64) = line.tν(ν; grad=true)

mutable struct ContinuousInterpolation
    ζ₋::Float64
    ζ₊::Float64
    lines::Vector{InterpolationThroughPML}
end

mutable struct Interpolation
    continuous_region::Vector{ContinuousInterpolation}
    rips::Vector{Rip}
end

function Interpolation(line::InterpolationThroughPML)
    Interpolation([ContinuousInterpolation(line.ζ, line.ζ, [line])], [])
end

import Base.push!
function push!(intrp::Interpolation, line::InterpolationThroughPML)
    push!(intrp.continuous_region[end].lines, line)
end

function push!(intrp::Interpolation, rip::Rip)
    push!(intrp.rips, rip)
    push!(intrp.continuous_region, ContinuousInterpolation(rip.ζ, rip.ζ, []))
end

function interpolate(u::AbstractFieldFunction, pml::PMLGeometry, ζs, ν_max)

    δ = 1e-1
    ε = 1e-4

    function create_line(ζ)
        ν_vec = Float64[]
        tν_vec = ComplexF64[]
        ∂tν_∂ν_vec = ComplexF64[]
        optimal_pml_transformation_solve(u, pml, ν_max, ζ, ν_vec, tν_vec, ∂tν_∂ν_vec)
        return InterpolationThroughPML(ζ, CubicHermiteSplineInterpolation(ν_vec, tν_vec, ∂tν_∂ν_vec))
    end

    function possible_rip_between(line1::InterpolationThroughPML, line2::InterpolationThroughPML)::Bool
        knots = 0:0.01:ν_max
        line1_points = line1.tν(knots)
        line2_points = line2.tν(knots)
        rel_diff = (2*norm(line1_points - line2_points)
            /(norm(line1_points) + norm(line2_points)))
        return rel_diff > δ
    end

    function recursively_subdivide(intrp, previous_line, next_line)
        ζ_mid = (previous_line.ζ + next_line.ζ)/2

        # We have to stop at some point and accept there's a rip
        if abs(previous_line.ζ - next_line.ζ) < ε
            push!(intrp, Rip(ζ_mid))
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

    return intrp
end
