

mutable struct BinaryAndQuadTreeInterpolation{T}
    coord::PMLCoordinates{1,T}
    bottom_left_corner_coord::PMLCoordinates{1,T}
    top_right_corner_coord::PMLCoordinates{1,T}
    tν::Complex{T} # Values of tν at centre of element
    ∇tν::Complex{T} # derivatives of tν at centre of element
    type::BinaryOrQuadNodeType
    top_left::TreeBasedInterpolation{T}
    top_right::TreeBasedInterpolation{T}
    bottom_right::TreeBasedInterpolation{T}
    bottom_left::TreeBasedInterpolation{T}
    QuadTreeBasedInterpolation(coord::PMLCoordinates{1,T}, tν, ∇tν) where {T} = new{T}(coord, complex(tν), complex(∇tν))
end


"""
    optimal_pml_transformation_solve!(field_fnc::Function, bqtree::BinaryAndQuadTreeInterpolation) -> bqtree

Find the optimal transformation for `field_fnc` on the domain defined by bqtree, by mutating bqtree
"""
function optimal_pml_transformation_solve!(field_fnc::Function, bqtree::BinaryAndQuadTreeInterpolation; ε=1e-12)

    ν_min = bqtree.bottom_left_corner_coord.ν
    ν_max = bqtree.top_right_corner_coord.ν
    ζ_min = bqtree.bottom_left_corner_coord.ζ
    ζ_max = bqtree.top_right_corner_coord.ζ

    if ν_min != 0 error("bottom left corner of bqtree must be at ν = 0") end

    while true

        # Find tν at centre of element

    end

    return bqtree
end
