
# uθ(x,y,θ,k) = exp(im*k*(x*cos(θ)+y*sin(θ)))

struct PlanarWaveSeries{N,T,M} <: AbstractFieldFunction
    modes::SVector{PlanarWave{N,T},M}
end

function planar_field(x,y,θ,k;θ_axis=π/2, p=[0,1], dx_dacross=[1,0])::Field
    darg_dx = [cos(θ+θ_axis), sin(θ+θ_axis)]
    darg_dtnu    = im*k*dot(p, darg_dx)
    darg_dacross = im*k*dot(dx_dacross, darg_dx)
    u = uθ(x,y,θ+θ_axis,k)
    return Field(u, darg_dtnu*u, darg_dacross*u, darg_dtnu^2*u, darg_dtnu*darg_dacross*u, darg_dtnu^3*u)
end

function two_planar_pole(x, y, x_pole, y_pole;k=4π, θ1=0.0, θ2=asin(2π/k), θ_axis=π/2, p=[0,1], dx_dacross=[1,0])::Field{Float64}

    A = 1 # Can be anything
    # Coefficient picked such that u/du_dx will be singular at x=x_pole, y=y_pole
    B = - A*sin(θ1+θ_axis)*uθ(x_pole,y_pole,θ1+θ_axis,k)/(sin(θ2+θ_axis)*uθ(x_pole,y_pole,θ2+θ_axis,k))

    return A*planar_field(x,y,θ1,k;θ_axis=θ_axis, p=p, dx_dacross=dx_dacross) + B*planar_field(x,y,θ2,k;θ_axis=θ_axis, p=p, dx_dacross=dx_dacross)

end

function two_planar_pole_location(x, x_pole, y_pole;k=1.0, θ1=0.0, θ2=asin(0.5), θ_axis=π/2, n=0)
    return ((x-x_pole)*(cos(θ1+θ_axis)-cos(θ2+θ_axis)) + (2π*n)/k)/(sin(θ2+θ_axis)-sin(θ1+θ_axis)) + y_pole
end
