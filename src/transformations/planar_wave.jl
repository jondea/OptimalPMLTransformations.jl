
function tν(planarwave::PlanarWave, pml::XAlignedRectangularPML, coords::PMLCoordinates)
    -im/planarwave.k * log(1-coords.ν)
end

function tν_jacobian(planarwave::PlanarWave, pml::XAlignedRectangularPML, coords::PMLCoordinates)
    im/(planarwave.k*(1-coords.ν))
end

function tν_and_jacobian(planarwave::PlanarWave, pml::XAlignedRectangularPML, coords::PMLCoordinates)
    tν(planarwave, pml, coords), tν_jacobian(planarwave, pml, coords)
end

function tx(planarwave::PlanarWave, pml::XAlignedRectangularPML, coords::PMLCoordinates)
    convert(CartesianCoordinates, PMLCoordinates(tν(planarwave, pml, coords), coords.ζ), pml)
end

function tx_and_jacobian(planarwave::PlanarWave, pml::XAlignedRectangularPML, coords::PMLCoordinates)
    _tν = tν(planarwave, pml, coords)
    _tx = convert(CartesianCoordinates, PMLCoordinates(_tν, coords.ζ), pml)

    tx_tν_jacobian = tx_tν_jacobian(pml) * tν_ν_jacobian(pml, _tν) * ν_x_jacobian(pml)

    _tx, tx_tν_jacobian
end

function tx_and_jacobian(planarwave::PlanarWave, pml::XAlignedRectangularPML, cartesian_coords::CartesianCoordinates)
    pml_coords = convert(PMLCoordinates, cartesian_coords)
    tx_and_jacobian(planarwave, pml, pml_coords)
end
