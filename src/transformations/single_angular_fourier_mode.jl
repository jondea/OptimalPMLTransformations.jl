
function tν(field::SingleAngularFourierMode, pml::AnnularPML, coords)
    return InverseHankelFunction.invhankelh1n(field.m, field.k*pml.R, 1-coords.ν/pml.δ)
end

function tν(u::SingleAngularFourierMode, pml::AnnularPML, coords::PMLCoordinates)
    -im/u.k * log(1-coords.ν)
end

function tν_jacobian(u::SingleAngularFourierMode, pml::AnnularPML, coords::PMLCoordinates)
    im/(u.k*(1-coords.ν))
end

function tν_and_jacobian(u::SingleAngularFourierMode, pml::AnnularPML, coords::PMLCoordinates)
    tν(u, pml, coords), tν_jacobian(u, pml, coords)
end

function tx(u::SingleAngularFourierMode, pml::AnnularPML, coords::PMLCoordinates)
    convert(CartesianCoordinates, PMLCoordinates(tν(u, pml, coords), coords.ζ), pml)
end

function tr_and_jacobian(u::SingleAngularFourierMode, pml::AnnularPML, coords::PMLCoordinates)

    _tν, _tν_jacobian = InverseHankelFunction.diffinvhankelh1n(field.m, field.k*pml.R, 1-coords.ν/pml.δ)

    # Is rt correct?
    pml.R + pml.δ*tν, _tν_jacobian
end

function tr_and_jacobian(u::SingleAngularFourierMode, pml::AnnularPML, polar_coords::PolarCoordinates)

    convert(PMLCoordinates, polar_coords, pml)

    _tν, _tν_jacobian = InverseHankelFunction.diffinvhankelh1n(field.m, field.k*pml.R, 1-pml_coords.ν/pml.δ)

    # Is rt correct?
    R + δ*tν, _tν_jacobian
end
