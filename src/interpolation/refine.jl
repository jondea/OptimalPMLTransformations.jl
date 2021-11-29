
function refine!(line::InterpLine, u::AbstractFieldFunction, pml::PMLGeometry)

    ζ = line.ζ

    extra_points = InterpPoint[]
    points = Base.Iterators.Stateful(line.points)
    prev_point = popfirst!(points)

    U_field = u(NamedTuple{(:u, :∂u_∂tν, :∂u_∂tζ, :∂2u_∂tν2, :∂2u_∂tν∂tζ, :∂3u_∂tν3)}, PMLCoordinates(0.0+0.0im,ζ), pml)
    field = U_field
    while !isempty(points)
        next_point = peek(points)

        ν = prev_point.ν
        tν = prev_point.tν
        ν_max = (prev_point.ν + next_point.ν)/2

        ν_vec = Float64[]
        tν_vec = ComplexF64[]
        ∂tν_∂ν_vec = ComplexF64[]
        ∂tν_∂ζ_vec = ComplexF64[]
        _, _, _, _, field = optimal_pml_transformation_solve(u, pml, ν_max, ζ, ν_vec, tν_vec, ∂tν_∂ν_vec, ∂tν_∂ζ_vec;ν0=ν, tν0=tν, field0=field, U_field)

        append!(extra_points, [InterpPoint(t...) for t in zip(ν_vec,tν_vec,∂tν_∂ν_vec,∂tν_∂ζ_vec)][2:end])
        prev_point = popfirst!(points)
    end

    # This costs nlogn, but could be done in n, by inserting into new array as we go along
    append!(line.points, extra_points)
    sort!(line.points, by=p->p.ν)
end

function refine_in_ζ!(region::ContinuousInterpolation, u::AbstractFieldFunction, pml::PMLGeometry)
    ν_max = 1.0
    function create_line(ζ)
        ν_vec = Float64[]
        tν_vec = ComplexF64[]
        ∂tν_∂ν_vec = ComplexF64[]
        ∂tν_∂ζ_vec = ComplexF64[]
        optimal_pml_transformation_solve(u, pml, ν_max, ζ, ν_vec, tν_vec, ∂tν_∂ν_vec, ∂tν_∂ζ_vec; silent_failure=true)
        # Add in point at ν=1? Try to work out if it's unbounded or not
        return InterpLine(ζ, ν_vec, tν_vec, ∂tν_∂ν_vec, ∂tν_∂ζ_vec)
    end

    extra_lines = InterpLine[]
    lines = Base.Iterators.Stateful(region.lines)
    prev_line = popfirst!(lines)

    while !isempty(lines)
        next_line = peek(lines)
        ζ = (prev_line.ζ + next_line.ζ)/2
        push!(extra_lines, create_line(ζ))
        prev_line = popfirst!(lines)
    end

    # This costs nlogn, but could be done in n, by inserting into new array as we go along
    append!(region.lines, extra_lines)
    sort!(region.lines, by=p->p.ζ)

end
