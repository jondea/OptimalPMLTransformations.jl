
function tν(planarwave::PlanarWave, pml::XAlignedRectangularPML, coords::PMLCoordinates)
    -im/planarwave.k[1] * log(1-coords.ν/pml.δ)
end

function tν_jacobian(planarwave::PlanarWave, pml::XAlignedRectangularPML, coords::PMLCoordinates)
    im/(planarwave.k[1]*(pml.δ-coords.ν))
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

    tx_x_jacobian = tx_tν_jacobian(pml) * tν_ν_jacobian(pml, tν_jacobian(planarwave, pml, coords)) * ν_x_jacobian(pml)

    _tx, tx_x_jacobian
end

function tx_and_jacobian(planarwave::PlanarWave, pml::XAlignedRectangularPML, cartesian_coords::CartesianCoordinates)
    pml_coords = convert(PMLCoordinates, cartesian_coords)
    tx_and_jacobian(planarwave, pml, pml_coords)
end
