@with_kw mutable struct Result
    assemble_time::Float64 = NaN
    solve_time::Float64 = NaN
    abs_sq_error::Float64 = NaN
    abs_sq_norm::Float64 = NaN
    rel_error::Float64 = NaN
end

csv_header(t) = csv_header(typeof(t))
csv_header(t::DataType) = join(string.(fieldnames(t)),',')

@generated function to_csv(p::P) where P
    ex = :(String[])
    for name in fieldnames(P)
        ex = :(vcat(string(p.$name), $ex))
    end
    return :(join($ex,','))
end
