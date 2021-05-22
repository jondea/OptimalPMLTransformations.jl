import Test: @test, @testset
using OptimalPMLTransformations
using OffsetArrays

@testset "Hankel series in annular PML" begin

    k = 2.0
    u = HankelSeries(k, OffsetVector([3.0+4.0im, 1.4+2.9im, 3.6 + 8.2im], -1:1) )

    R = 3.0
    δ = 1.0

    pml = AnnularPML(R, δ)

    ζ = 0.2
    ν = 0.9
    u_tν(tν) = u(NamedTuple{(:u, :∂u_∂tν, :∂u_∂tζ, :∂2u_∂tν2, :∂3u_∂tν3)}, PMLCoordinates(tν,ζ), pml)
    tν, ∂tν_∂ν, ∂tν_∂ζ = optimal_pml_transformation_solve(u_tν, ν; householder_order=1)

    U = u_tν(0).u

    @test u_tν(tν).u ≈ U*(1-ν)
end
