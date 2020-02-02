
mutable struct BinaryAndQuadTreeInterpolation{T}
    coord::PMLCoordinates{1,T}
    bottom_left_corner_coord::PMLCoordinates{1,T}
    top_right_corner_coord::PMLCoordinates{1,T}
    tν::Complex{T} # Values of tν at centre of element
    ∇tν::Complex{T} # derivatives of tν at centre of element
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
    bqtree.bottom_right = bottom_bqtree
    bqtree.top_right = top_bqtree
    bqtree.top_left = top_bqtree
end

function split_across!(bqtree::BinaryAndQuadTreeInterpolation,
                        left_bqtree::BinaryAndQuadTreeInterpolation,
                        right_bqtree::BinaryAndQuadTreeInterpolation)
    bqtree.bottom_left = left_bqtree
    bqtree.top_left = left_bqtree
    bqtree.bottom_right = right_bqtree
    bqtree.top_right = right_bqtree
end

function (bqtree::BinaryAndQuadTreeInterpolation)(coord::PMLCoordinates)::Tuple{Complex{T}, Complex{T}} where {T}

    # Note: we bias towards the bottom left if it is on the boundary
    if coord.ν > bqtree.coord.ν
        if coord.ζ[1] > bqtree.coord.ζ[1]
            # Top Right
            if isassigned(bqtree.top_right)
                return bqtree.top_right(coord)
            end

            if coord.ν > bqtree.top_right_corner_coord.ν throw(DomainError(coord.ν, "Coordinate is outside of domain of interpolation")) end
            if coord.ζ > bqtree.top_right_corner_coord.ζ[1] throw(DomainError(coord.ζ[1]), "Coordinate is outside of domain of interpolation") end

            return interpolate(bqtree, coord)
        else
            # Top Left
            if isassigned(bqtree.top_left)
                return bqtree.top_right(coord)
            end

            if coord.ν > bqtree.top_right_corner_coord.ν throw(DomainError(coord.ν, "Coordinate is outside of domain of interpolation")) end
            if coord.ζ < bqtree.bottom_left_corner_coord.ζ[1] throw(DomainError(coord.ζ[1]), "Coordinate is outside of domain of interpolation") end

            return interpolate(bqtree, coord)
        end
    else
        if coord.ζ[1] > bqtree.coord.ζ[1]
            # Bottom Right
            if isassigned(bqtree.bottom_right)
                return bqtree.top_right(coord)
            end

            if coord.ν < bqtree.bottom_left_corner_coord.ν throw(DomainError(coord.ν, "Coordinate is outside of domain of interpolation")) end
            if coord.ζ > bqtree.top_right_corner_coord.ζ[1] throw(DomainError(coord.ζ[1]), "Coordinate is outside of domain of interpolation") end

            return interpolate(bqtree, coord)
        else
            # Bottom Left
            if isassigned(bqtree.bottom_left)
                return bqtree.bottom_left(coord)
            end

            if coord.ν < bqtree.bottom_left_corner_coord.ν throw(DomainError(coord.ν, "Coordinate is outside of domain of interpolation")) end
            if coord.ζ < bqtree.bottom_left_corner_coord.ζ[1] throw(DomainError(coord.ζ[1]), "Coordinate is outside of domain of interpolation") end

            return interpolate(bqtree, coord)
        end
    end
end

"""
    interpolate(bqtree::BinaryAndQuadTreeInterpolation, coord::PMLCoordinates) = (tν::Complex, ∇tν::Complex)

Blindly interpolate tν and ∇tν to coord using the bqtree
"""
function interpolate(bqtree::BinaryAndQuadTreeInterpolation{T}, coord::PMLCoordinates{1,T})::Tuple{Complex{T}, Complex{T}} where {T}
    tν = bqtree.tν # + taylor series
    ∇tν = bqtree.∇tν # + taylor series?
    return (tν, ∇tν)
end
