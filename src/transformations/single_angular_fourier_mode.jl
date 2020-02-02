

function single_pml_opt_trans(m, kR, pml_coords::PMLCoordinates)
    return invnormalisedhankelh1(m, kR, 1-pml_coords.ν)
end

function single_pml_opt_trans(m, kR, ν::Number)
    return invnormalisedhankelh1(m, kR, 1-ν)
end

function optimal_pml_transformation(field::SingleAngularFourierMode, pml::AnnularPML)
    coords -> invnormalisedhankelh1(field.m, field.k*pml.R, coords)
end

function optimal_pml_transformation(field::SingleAngularFourierMode, pml::AnnularPML, coords)
    return optimal_pml_transformation(field, pml)(coords)
end
