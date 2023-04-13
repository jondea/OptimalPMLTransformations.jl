
struct InterpPatchIterator
	intrp0::InterpLine
	intrp1::InterpLine
end

"""
    eachpatch(line0::InterpLine, line1::InterpLine)

Return an iterator over patches formed by the points of two lines.
"""
function eachpatch(line0::InterpLine, line1::InterpLine)
    # We could probably deal with this by extrapolation, but we shouldn't have to
    @assert first(line0.points).ν == first(line1.points).ν
    @assert last(line0.points).ν == last(line1.points).ν

	InterpPatchIterator(line0, line1)
end

Base.IteratorSize(::InterpPatchIterator) = Base.SizeUnknown()

Base.IteratorEltype(::InterpPatchIterator) = Base.HasEltype()

Base.eltype(::InterpPatchIterator) = InterpPatch

function Base.iterate(it::InterpPatchIterator, state)

	if isempty(state.intrp_points0) || isempty(state.intrp_points1)
		return nothing
	end

	# If one is smaller than the other, use the point with the smallest ν,
	# popping it to show it has been used. Interpolate by peeking forward for the other
	if peek(state.intrp_points0).ν > peek(state.intrp_points1).ν
		intrp11 = popfirst!(state.intrp_points1)
		intrp10 = InterpPoint(evaluate(InterpSegment(state.intrp00, peek(state.intrp_points0), it.intrp0.ζ), intrp11.ν))
	elseif peek(state.intrp_points1).ν > peek(state.intrp_points0).ν
		intrp10 = popfirst!(state.intrp_points0)
		intrp11 = InterpPoint(evaluate(InterpSegment(state.intrp01, peek(state.intrp_points1), it.intrp0.ζ), intrp10.ν))
	else # Equal, use both
		intrp10 = popfirst!(state.intrp_points0)
		intrp11 = popfirst!(state.intrp_points1)
	end

	patch = InterpPatch(state.intrp00.ν, intrp11.ν, it.intrp0.ζ, it.intrp1.ζ,
                        Dtν_νζ(state.intrp00), Dtν_νζ(state.intrp01), Dtν_νζ(intrp10), Dtν_νζ(intrp11))

	(patch, (;state.intrp_points0, state.intrp_points1, intrp00=intrp10, intrp01=intrp11))
end

function Base.iterate(it::InterpPatchIterator)
	intrp_points0 = Base.Iterators.Stateful(it.intrp0.points)
    intrp_points1 = Base.Iterators.Stateful(it.intrp1.points)

	if isempty(intrp_points0) || isempty(intrp_points1)
		return nothing
	end

	intrp00 = popfirst!(intrp_points0)
    intrp01 = popfirst!(intrp_points1)

	# Set up initial state then delegate to stateful iterate function
	Base.iterate(it, (;intrp_points0, intrp_points1, intrp00, intrp01))
end
