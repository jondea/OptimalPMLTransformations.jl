"""
    InvHankelPML(geom <: AnnularPML, mode <: SingleAngularFourierMode, cache)

An optimal transformation defined on an annular PML which is the inverse of
single angular Fourier/radial Hankel mode, functionally this is equivalent to
the inverse of a Hankel function. By default this struct has a simple cache,
which greatly speeds up evaluation on regularly FEM grids (a typical use case).
The cache can be disabled by passing in `nothing`.
"""
mutable struct InvHankelPML{P <: AnnularPML, M <: SingleAngularFourierMode,C}
    "Geometry of the PML on which the transformation is defined"
    geom::P
    "Single mode we will invert to find the transformation"
    mode::M
    "Cache the values and derivatives of the transformation as we evaluate them"
    cache::C
end

InvHankelPML(;R,δ,k,m) = InvHankelPML(AnnularPML(R,δ), SingleAngularFourierMode(k,m,R), Dict{typeof(k),NTuple{2,ComplexF64}}())

function tν(field::SingleAngularFourierMode, pml::AnnularPML, coords)
    return invhankelh1n(field.m, field.k*pml.R, 1-coords.ν/pml.δ)
end

function tx(u::SingleAngularFourierMode, pml::AnnularPML, coords::PMLCoordinates)
    convert(CartesianCoordinates, PMLCoordinates(tν(u, pml, coords), coords.ζ), pml)
end

function tr(u::SingleAngularFourierMode, pml::AnnularPML, coords::PMLCoordinates)
    _tν = invhankelh1n(u.m, u.k*pml.R, 1-coords.ν/pml.δ)
    _tr = pml.δ*_tν/u.k
    return _tr
end

function tr_and_jacobian(t::InvHankelPML, coords)
    tr_and_jacobian(t.mode, t.geom, coords; cache=t.cache)
end

function tr_and_jacobian(u::SingleAngularFourierMode, pml::AnnularPML, coords::PMLCoordinates; cache = nothing)
    ν = coords.ν
    R, δ = pml.R, pml.δ
    m, k = u.m, u.k
    if ν <= 0
        return ComplexF32(pml.R + ν), SDiagonal{2,ComplexF32}((1, 1))
    elseif ν >= 1
        return NaN + NaN*im, SDiagonal{2,ComplexF32}((NaN + NaN*im, NaN + NaN*im))
    end
    if !isnothing(cache)
        # Identical knot points may vary a bit, so we round to bunch them up.
        # invhankel has a tolerance anyway, so it shouldn't make much
        # difference..
        ν_key = round(ν; sigdigits=14)
        if ν_key ∈ keys(cache)
            _tν, _tν_jacobian = cache[ν_key]
        else
            _tν, _tν_jacobian = diffinvhankelh1n(m, k*R, 1-ν_key/δ)
            cache[ν_key] = (_tν, _tν_jacobian)
        end
    else
        _tν, _tν_jacobian = diffinvhankelh1n(m, k*R, 1-ν/δ)
    end
    _tr = δ*_tν/k
    ∂tr_∂r = -δ*_tν_jacobian/k
    return _tr, SDiagonal(∂tr_∂r, 1)
end

function tr_and_jacobian(u::SingleAngularFourierMode, pml::AnnularPML, polar_coords::PolarCoordinates; cache=nothing)
    pml_coords = convert(PMLCoordinates, polar_coords, pml)
    return tr_and_jacobian(u, pml, pml_coords; cache)
end
