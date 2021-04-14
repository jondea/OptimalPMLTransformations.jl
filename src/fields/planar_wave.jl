
struct PlanarWave{N,T} <: AbstractFieldFunction
    k::SVector{N,T}
    a::T
end

PlanarWave(k,a=one(first(k))) = PlanarWave(SVector(k),a)
PlanarWave((k,θ)::NamedTuple{(:k, :θ)},a=one(first(k))) = PlanarWave{2}(SVector(k*cos(θ), k*sin(θ)),a)

(planarwave::PlanarWave)(coords::CartesianCoordinates) = planarwave.a *exp(im*dot(coords.x, planarwave.k))

"Tensor contraction of two vectors"
contract(x::AbstractVector, y::AbstractVector) = mapreduce(*, +, x, y)

function (planarwave::PlanarWave)(::FieldAndDerivativesAtPoint, coords::PMLCoordinates, geom::PMLGeometry)
    k = planarwave.k

    u = f(convert(CartesianCoordinates, coords, geom))
    @einsum du_dx[i] := im * k[i] * u
    @einsum d2u_dxdx[i,j] := im * im * k[i] * k[j] * u
    @einsum d2u_dxdxdx[i,j,m] = im * im * im * k[i] * k[j] * k[m] * u

    dx_dtν, dx_dζ, d2x_dtνdζ, d2x_dtν2, d3x_dtν3 = geom(PMLGeometryDerivatives(), )

    # return Field(u, darg_dtnu*u, darg_dacross*u, darg_dtnu^2*u, darg_dtnu*darg_dacross*u, darg_dtnu^3*u)

    @einsum du_dtν := du_dx[i]*dx_dtν[i]
    @einsum du_dζ := du_dx[i]*dx_dζ[i]
    @einsum d2u_dtν2 := d2u_dxdx[i,j]*dx_dtν[i]*dx_dtν[j] + du_dx[i]*d2x_dtν2[i]
    @einsum d2u_dtνdζ := d2u_dxdx[i,j]*dx_dtν[i]*dx_dζ[j] + du_dx[i]*d2x_dtνdζ[i]
    @einsum d3u_dtν3 := (d2u_dxdxdx[i,j,m]*dx_dtν[i]*dx_dtν[j]*dx_dtν[m] + 2*d2u_dxdx[i,j]*d2x_dtν2[i]*dx_dtν[j]
                        + d2u_dxdx[i,j]*d2x_dtν2[i]*dx_dtν[j] + du_dx[i]*d3x_dtν3[i])

    return FieldAndDerivativesAtPoint(u, du_dtν, du_dζ, d2u_dtν2, d2u_dtνdζ, d3u_dtν3)
end
