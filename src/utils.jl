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
function padded_range(a, n::Integer)
    (minimum(a)-n):(maximum(a)+n)
end

consecutive_pairs(r) = partition(r, 2, 1)

struct OnlineKBN{T}
    "Sum"
    s::T
    "Correction"
    c::T
end

OnlineKBN(T::DataType) = OnlineKBN(Base.reduce_empty(+, T),Base.reduce_empty(+, T))

Base.sum(online_kbn::OnlineKBN) = online_kbn.s

function Base.:+(kbn::OnlineKBN, b)
    s, c = online_sum_kbn(kbn.s, b, kbn.c)
    return OnlineKBN(s, c)
end

Base.:-(kbn::OnlineKBN, a) = Base.:+(kbn::OnlineKBN, -a)

function online_sum_kbn(a::T, b::T, c::T = Base.reduce_empty(+, T)) where {T}
    s = a - c
    t = s + b
    if abs(s) >= abs(b)
        c -= ((s-t) + b)
    else
        c -= ((b-t) + s)
    end
    s = t
    return s - c, c
end
