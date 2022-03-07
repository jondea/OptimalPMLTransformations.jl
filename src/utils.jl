"""
    insertsorted!(v, el; by=identity)

Insert `el` into a sorted `v` at the correct sorted index.
"""
function insertsorted!(v, el; by=identity)
    i = searchsortedfirst(v, el; by)
    insert!(v, i, el)
end

"""
    padded_range(a, n)

Return range with `n` added to beginning and end.
"""
function padded_range(a, n::Integer) where T
    (minimum(a)-n):(maximum(a)+n)
end
