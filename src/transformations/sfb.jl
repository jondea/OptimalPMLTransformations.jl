
struct SFB{G <: PMLGeometry}
    geom::G
    k::Float64
end

function tν(t::SFB, coords::PMLCoordinates)
    ν, k, δ = coords.ν, t.k, t.geom.δ
    return - im/k*log(1-ν/δ)
end

function ∂tν_∂ν(t::SFB, coords::PMLCoordinates)
    ν, k, δ = coords.ν, t.k, t.geom.δ
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

