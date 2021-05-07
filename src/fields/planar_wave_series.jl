
# uθ(x,y,θ,k) = exp(im*k*(x*cos(θ)+y*sin(θ)))

struct PlanarWaveSeries{M,N,KT,AT} <: AbstractFieldFunction
    modes::SVector{M,PlanarWave{N,KT,AT}}
end

import Base: +
+(u1::PlanarWave{N,KT,AT}, u2::PlanarWave{N,KT,AT}) where {N,KT,AT} = PlanarWaveSeries(SVector(u1,u2))
+(u1::PlanarWaveSeries, u2::PlanarWave) = PlanarWaveSeries(vcat(u1.modes,SVector(u2)))
+(u1::PlanarWave, u2::PlanarWaveSeries) = u2 + u1

(u::PlanarWaveSeries{M,N,KT,AT})(coords::CartesianCoordinates) where {M,N,KT,AT} = mapreduce(m->(m)(coords), +, u.modes; init=zero(AT))


function two_planar_pole(x, y, x_pole, y_pole;k=4π, θ1=0.0, θ2=asin(2π/k), θ_axis=π/2)

    A = 1 # Can be anything
    # Coefficient picked such that u/du_dx will be singular at x=x_pole, y=y_pole
    B = - A*sin(θ1+θ_axis)*uθ(x_pole,y_pole,θ1+θ_axis,k)/(sin(θ2+θ_axis)*uθ(x_pole,y_pole,θ2+θ_axis,k))

    return A*planar_field(θ1,k) + B*planar_field(x,y,θ2,k;θ_axis=θ_axis, p=p, dx_dacross=dx_dacross)

end

function two_planar_pole_location(x, x_pole, y_pole;k=1.0, θ1=0.0, θ2=asin(0.5), θ_axis=π/2, n=0)
    return ((x-x_pole)*(cos(θ1+θ_axis)-cos(θ2+θ_axis)) + (2π*n)/k)/(sin(θ2+θ_axis)-sin(θ1+θ_axis)) + y_pole
end
