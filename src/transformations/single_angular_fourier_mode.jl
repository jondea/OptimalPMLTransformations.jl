
struct InvHankelPML{P <: AnnularPML, M <: SingleAngularFourierMode}
    "Geometry of the PML"
    geom::P
    "Wavenumber of the equation/field"
    mode::M
end

InvHankelPML(;R,δ,k,m) = InvHankelPML(AnnularPML(R,δ), SingleAngularFourierMode(k,m,R))

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
    tr_and_jacobian(t.mode, t.geom, coords)
end

function tr_and_jacobian(u::SingleAngularFourierMode, pml::AnnularPML, coords::PMLCoordinates)
    ν = coords.ν
    R, δ = pml.R, pml.δ
    m, k = u.m, u.k
    if ν <= 0
        return ComplexF32(pml.R + ν), SDiagonal{2,ComplexF32}((1, 1))
    elseif ν >= 1
        return NaN + NaN*im, SDiagonal{2,ComplexF32}((NaN + NaN*im, NaN + NaN*im))
    end
    _tν, _tν_jacobian = diffinvhankelh1n(m, k*R, 1-ν/δ)
    _tr = δ*_tν/k
    ∂tr_∂r = -δ*_tν_jacobian/k
    return _tr, SDiagonal(∂tr_∂r, 1)
end

function tr_and_jacobian(u::SingleAngularFourierMode, pml::AnnularPML, polar_coords::PolarCoordinates)
    pml_coords = convert(PMLCoordinates, polar_coords, pml)
    return tr_and_jacobian(u, pml, pml_coords)
end
