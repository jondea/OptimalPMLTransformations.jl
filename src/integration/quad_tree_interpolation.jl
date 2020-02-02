
mutable struct QuadTreeInterpolation{T}
    coord::PMLCoordinates{1,T}
    tν::Complex{T} # Values of tν at centre of element
    ∇tν::Complex{T} # derivatives of tν at centre of element
    top_left::TreeBasedInterpolation{T}
    top_right::TreeBasedInterpolation{T}
    bottom_right::TreeBasedInterpolation{T}
    bottom_left::TreeBasedInterpolation{T}
    QuadTreeBasedInterpolation(coord::PMLCoordinates{1,T}, tν, ∇tν) where {T} = new{T}(coord, complex(tν), complex(∇tν))
end

function (interp::QuadTreeInterpolation)(coord::PMLCoordinates)
    
    if (coord == interp.coord)
        return (tν, ∇tν)
    end
    
    # Evaluate recursively
    if (coord.ν > interp.coord.ν)
        ifinterp.top_left()
    end
    
    return ()
end