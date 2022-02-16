struct Rip
    ζ::Float64
end

struct InterpPoint
    ν::Float64
    tν::ComplexF64
    ∂tν_∂ν::ComplexF64
    ∂tν_∂ζ::ComplexF64
end

import Base.zero
zero(::Type{InterpPoint}) = InterpPoint(0, 0, 0, 0)

struct InterpLine
    ζ::Float64
    points::Vector{InterpPoint}
end

struct InterpPatch
    p00::InterpPoint
    p01::InterpPoint
    p10::InterpPoint
    p11::InterpPoint
    ζ0::Float64
    ζ1::Float64
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

import Base.summary
function summary(io::IO, intrp::Interpolation)
    println(io, "Interpolation with $(length(intrp.rips)) rip$(length(intrp.rips) != 1 ? "s" : "")")
    if !isempty(intrp.rips)
        println(io, "Rips:")
        for rip in intrp.rips
            println(io, "    ζ = $(rip.ζ)")
        end
    end
    if !isempty(intrp.continuous_region)
        println(io, "Continuous regions of interpolation:")
        for region in intrp.continuous_region
            nlines = length(region.lines)
            println(io, "    ζ from $(region.ζ₋) to $(region.ζ₊) with $(nlines) line$(nlines != 1 ? "s" : "")")
        end
    end
end

import Base.show
show(io::IO, ::MIME"text/plain", i::Interpolation) = summary(io, i)

function Interpolation(line::InterpLine)
    Interpolation([ContinuousInterpolation(line.ζ, line.ζ, [line])], [])
end

import Base.push!
function push!(intrp::Interpolation, line::InterpLine)
    push!(intrp.continuous_region[end].lines, line)
end

function push!(intrp::Interpolation, rip::Rip)
    last(intrp.continuous_region).ζ₊ = rip.ζ
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

import Base.isnan
isnan(p::InterpPoint) = isnan(p.ν) || isnan(p.tν) || isnan(p.∂tν_∂ν) || isnan(p.∂tν_∂ζ)
isnan(p::Dtν_νζ) = isnan(p.tν) || isnan(p.∂tν_∂ν) || isnan(p.∂tν_∂ζ)
isnan(p::Dtν_ν) = isnan(p.tν) || isnan(p.∂tν_∂ν)

import Base.isinf
isinf(p::InterpPoint) = isinf(p.ν) || isinf(p.tν) || isinf(p.∂tν_∂ν) || isinf(p.∂tν_∂ζ)
isinf(p::Dtν_νζ) = isinf(p.tν) || isinf(p.∂tν_∂ν) || isinf(p.∂tν_∂ζ)
isinf(p::Dtν_ν) = isinf(p.tν) || isinf(p.∂tν_∂ν)
