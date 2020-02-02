

optimal_transformed_pml_coordinates(planarwave::PlanarWave, pml::XAlignedRectangularPML) = ν -> -im/planarwave.k * log(1-ν)

optimal_transformed_cartesian_coordinates(planarwave::PlanarWave, pml::XAlignedRectangularPML) = ν -> -im/planarwave.k * log(1-ν)

function optimal_transformed_pml_coordinates(planarwave::PlanarWave, pml::XAlignedRectangularPML, pml_coords::PMLCoordinates)
    return optimal_pml_transformation(planarwave, pml)(pml_coords)
end
