"""
SFB(geom, k)

Scale free Bermudez (SFB) transformation defined by the PML geometry (`geom`)
and wavenumber (`k`).

Scale free Bermudez is a PML transformation proposed in Deakin's PhD
thesis. It is a slight modification to one proposed by Bermudez et al (2007),
removing the dependency of the PML thickness in the solution. The transformed
through-the-PML coordinate is
```
tν = -im/k*log(1-ν/δ)
```
where `δ` is the PML thickness.
"""
struct SFB{G <: PMLGeometry}
    "Geometry of the PML"
    geom::G
    "Wavenumber of the equation/field"
    k::Float64
end

function tν(t::SFB, coords::PMLCoordinates)
    ν, k, δ = coords.ν, t.k, pml_thickness(t.geom)
    return -im/k*log(1-ν/δ)
end

function ∂tν_∂ν(t::SFB, coords::PMLCoordinates)
    ν, k, δ = coords.ν, t.k, pml_thickness(t.geom)
    return im/(k*δ)/(1-ν)
end

function tr(t::SFB{<:AnnularPML}, coords::PMLCoordinates)
    R = t.geom.R
    return R + tν(t, coords)
end

function tr(t::SFB{<:AnnularPML}, coords::PolarCoordinates)
    tr(t, convert(PMLCoordinates,coords,t.geom))
end

function ∂tr_∂r(t::SFB{<:AnnularPML}, coords::PMLCoordinates)
    ∂tν_∂ν(t, coords)
end

function ∂tr_∂r(t::SFB{<:AnnularPML}, coords::PolarCoordinates)
    ∂tν_∂ν(t, convert(PMLCoordinates,coords,t.geom))
end

jacobian(t::SFB{<:AnnularPML}, coords::PolarCoordinates) = SDiagonal{2}((∂tr_∂r(t,coords), 1))
tr_and_jacobian(t::SFB{<:AnnularPML}, coords::PolarCoordinates) = tr(t,coords), jacobian(t,coords)

function tr_and_jacobian(t::SFB{<:AnnularPML}, x::CartesianCoordinates)
    p = convert(PolarCoordinates, x)
    r, θ = p.r, p.θ
    sinθ, cosθ  = sincos(θ)
    tr_ = tr(t, p)
    jacobian_tx_tr = SMatrix{2,2}([
        cosθ  -sinθ/r;
        sinθ   cosθ/r
    ])
    jacobian_tr_r = jacobian(t,p)
    jacobian_r_x = SMatrix{2,2}([
        cosθ       sinθ;
        -tr_ *sinθ tr_*cosθ
    ])
    return jacobian_tx_tr*jacobian_tr_r*jacobian_r_x, tr_
end

jacobian(t::SFB{<:AnnularPML}, x::CartesianCoordinates) = last(tr_and_jacobian(t,x))

