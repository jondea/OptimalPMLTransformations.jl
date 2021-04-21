import Test: @test, @testset
using OptimalPMLTransformations

@testset "Single planar wave" begin

    k = 3.2
    θ = 0.3
    a = 3.4
    u = PlanarWave((;k,θ),a)

    @test u(CartesianCoordinates(0,0)) ≈ a

    pml = XAlignedRectangularPML(1.0, 2.0)

    ν = 2.7
    ζ = 4.8
    U = u(PMLCoordinates(0.0, ζ))
    ν_bar = ν/pml.δ
    tx, J = tx(u, pml, PMLCoordinates(ν, ζ))
    @test u() ≈ U*(1-ν_bar)
    ε = 1e-8
    @test (tx(u, pml, PMLCoordinates(ν+ε, ζ)) - tx)/ε ≈ J[:,1]
    @test (tx(u, pml, PMLCoordinates(ν, ζ+ε)) - tx)/ε ≈ J[:,2]

end

@testset "Sum of planar waves" begin

    PlanarWave(k,a=one(first(k))) = PlanarWave(SVector(k),a)
    PlanarWave((k,θ)::NamedTuple{(:k, :θ)},a=one(first(k))) = PlanarWave{2}(SVector(k*cos(θ), k*sin(θ)),a)

    function tx_and_jacobian(planarwave::PlanarWave, pml::XAlignedRectangularPML, cartesian_coords::CartesianCoordinates)

end
