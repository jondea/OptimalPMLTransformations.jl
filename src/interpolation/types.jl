struct Rip
    ζ::Float64
end

struct InterpPoint
    ν::Float64
    tν::ComplexF64
    ∂tν_∂ν::ComplexF64
    ∂tν_∂ζ::ComplexF64
end

struct InterpLine
    ζ::Float64
    points::Vector{InterpPoint}
end

function InterpLine(ζ::Float64, ν_vec::Vector{Float64}, tν_vec::Vector{ComplexF64}, ∂tν_∂ν_vec::Vector{ComplexF64}, ∂tν_∂ζ_vec::Vector{ComplexF64})
    InterpLine(ζ, [InterpPoint(t...) for t in zip(ν_vec,tν_vec,∂tν_∂ν_vec,∂tν_∂ζ_vec)])
end

mutable struct ContinuousInterpolation
    ζ₋::Float64
    ζ₊::Float64
    lines::Vector{InterpLine}
end

mutable struct Interpolation
    continuous_region::Vector{ContinuousInterpolation}
    rips::Vector{Rip}
end

function Interpolation(line::InterpLine)
    Interpolation([ContinuousInterpolation(line.ζ, line.ζ, [line])], [])
end

import Base.push!
function push!(intrp::Interpolation, line::InterpLine)
    push!(intrp.continuous_region[end].lines, line)
end

function push!(intrp::Interpolation, rip::Rip)
    push!(intrp.rips, rip)
    push!(intrp.continuous_region, ContinuousInterpolation(rip.ζ, rip.ζ, []))
end

struct Dtν_ν
    tν::ComplexF64
    ∂tν_∂ν::ComplexF64
end

Dtν_ν(p::InterpPoint) = Dtν_νζ(p.tν, p.∂tν_∂ν)

struct Dtν_νζ
    tν::ComplexF64
    ∂tν_∂ν::ComplexF64
    ∂tν_∂ζ::ComplexF64
end

Dtν_νζ(p::InterpPoint) = Dtν_νζ(p.tν, p.∂tν_∂ν, p.∂tν_∂ζ)

InterpPoint(ν::Float64, d::Dtν_νζ) = InterpPoint(ν, d.tν, d.∂tν_∂ν, d.∂tν_∂ζ)