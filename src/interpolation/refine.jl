
function refine!(line::InterpLine, u::AbstractFieldFunction, pml::PMLGeometry)

    ζ = line.ζ

    extra_points = InterpPoint[]
    points = Base.Iterators.Stateful(line.points)
    prev_point = popfirst!(points)

    U_field = u(pml, zero(tν), NamedTuple{(:u, ∂tν_∂ζ)})
    field =
    while !isempty(points)
        next_point = peek(points)

        ν = prev_point.ν
        tν = prev_point.tν
        ν_max = next_point

        ν_vec = Float64[]
        tν_vec = ComplexF64[]
        ∂tν_∂ν_vec = ComplexF64[]
        ∂tν_∂ζ_vec = ComplexF64[]
        _, _, _, _, field = optimal_pml_transformation_solve(u, pml, ν_max, ζ, ν_vec, tν_vec, ∂tν_∂ν_vec, ∂tν_∂ζ_vec, ν, tν, field, U_field)

        append!(extra_points, [InterpPoint(t...) for t in zip(ν_vec,tν_vec,∂tν_∂ν_vec,∂tν_∂ζ_vec)][2:end])
        prev_point = popfirst!(points)
    end

    append!(line.points, extra_points)
    sort!(line.points, by=p->p.ν)
end
