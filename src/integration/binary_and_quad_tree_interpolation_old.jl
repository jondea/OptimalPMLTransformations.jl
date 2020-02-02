
@enum BinaryOrQuadNodeType begin
    BINARY_THROUGH # Binary (2-way) split in the ν direction
    BINARY_ACROSS # Binary (2-way) split in the ζ direction
    QUAD # Four way split (2 in each direction)
    LEAF # No further splitting, this is the last node in the tree
end

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

function split_through!(bqtree::BinaryAndQuadTreeInterpolation,
                        top_bqtree::BinaryAndQuadTreeInterpolation,
                        bottom_bqtree::BinaryAndQuadTreeInterpolation)
    bqtree.bottom_left = bottom_bqtree
    bqtree.top_right = top_bqtree
    bqtree.type = BINARY_THROUGH
end

bottom(bqtree::BinaryAndQuadTreeInterpolation) = bqtree.bottom_left
top(bqtree::BinaryAndQuadTreeInterpolation) = bqtree.top_right

function split_across!(bqtree::BinaryAndQuadTreeInterpolation,
                        left_bqtree::BinaryAndQuadTreeInterpolation,
                        right_bqtree::BinaryAndQuadTreeInterpolation)
    bqtree.top_left = left_bqtree
    bqtree.bottom_right = right_bqtree
    bqtree.type = BINARY_THROUGH
end

left(bqtree::BinaryAndQuadTreeInterpolation) = bqtree.top_left
right(bqtree::BinaryAndQuadTreeInterpolation) = bqtree.bottom_right

function (bqtree::BinaryAndQuadTreeInterpolation)(coord::PMLCoordinates)
    
    if bqtree.type == QUAD
        if coord.ν > bqtree.coord.ν
            if coord.ζ[1] > bqtree.coord.ζ[1]
                if !isassigned(bqtree.top_right) error("Malformed tree: type of node is QUAD by top_right is not assigned") end
                if coord.ν > bqtree.top_right_corner_coord.ν throw(DomainError(coord.ν, "Coordinate is outside of domain of interpolation")) end
                if coord.ζ > bqtree.top_right_corner_coord.ζ[1] throw(DomainError(coord.ζ[1]), "Coordinate is outside of domain of interpolation") end
                bqtree.top_right(coord)
            else
            end
        else
    elseif bqtree.type == BINARY_THROUGH
    elseif bqtree.type == BINARY_ACROSS
    else
        return (tν, ∇tν)
    end
    
end
