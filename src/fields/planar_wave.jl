
struct PlanarWave{N,KT,AT} <: AbstractFieldFunction
    k::SVector{N,KT}
    a::AT
    function PlanarWave(k::Union{AbstractVector,Tuple},a=one(first(k)))
        k_svec = SVector(k)
        new{length(k_svec), eltype(k_svec), typeof(a)}(k_svec,a)
    end
end

function PlanarWave(;k, θ=zero(k), a=one(k))
    PlanarWave(SVector(k*cos(θ), k*sin(θ)),a)
end

import Base: *
*(a::Number, u::PlanarWave) = PlanarWave(u.k, a*u.a)

(planarwave::PlanarWave)(coords::CartesianCoordinates) = planarwave.a * exp(im*contract(coords.x, planarwave.k))

function (planarwave::PlanarWave)(::FieldAndDerivativesAtPoint, coords::PMLCoordinates, geom::PMLGeometry)
    k = planarwave.k

    u = f(convert(CartesianCoordinates, coords, geom))
    @einsum du_dx[i] := im * k[i] * u
    @einsum d2u_dxdx[i,j] := im * im * k[i] * k[j] * u
    @einsum d2u_dxdxdx[i,j,m] = im * im * im * k[i] * k[j] * k[m] * u

    dx_dtν, dx_dζ, d2x_dtνdζ, d2x_dtν2, d3x_dtν3 = geom(PMLGeometryDerivatives(), )

    # return Field(u, darg_dtnu*u, darg_dacross*u, darg_dtnu^2*u, darg_dtnu*darg_dacross*u, darg_dtnu^3*u)

    @einsum ∂u_∂tν := du_dx[i]*dx_dtν[i]
    @einsum ∂u_∂tζ := du_dx[i]*dx_dζ[i]
    @einsum ∂2u_∂tν2 := d2u_dxdx[i,j]*dx_dtν[i]*dx_dtν[j] + du_dx[i]*d2x_dtν2[i]
    @einsum d2u_dtνdζ := d2u_dxdx[i,j]*dx_dtν[i]*dx_dζ[j] + du_dx[i]*d2x_dtνdζ[i]
    @einsum ∂3u_∂tν3 := (d2u_dxdxdx[i,j,m]*dx_dtν[i]*dx_dtν[j]*dx_dtν[m] + 2*d2u_dxdx[i,j]*d2x_dtν2[i]*dx_dtν[j]
                        + d2u_dxdx[i,j]*d2x_dtν2[i]*dx_dtν[j] + du_dx[i]*d3x_dtν3[i])

    return FieldAndDerivativesAtPoint(u, ∂u_∂tν, ∂u_∂tζ, ∂2u_∂tν2, d2u_dtνdζ, ∂3u_∂tν3)
end
