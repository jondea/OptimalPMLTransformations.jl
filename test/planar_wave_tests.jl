import Test: @test, @testset
using OptimalPMLTransformations

@testset "Planar wave basics" begin
    k = 3.2
    θ = 0.3
    a = 3.4
    u1 = PlanarWave((k*cos(θ), k*sin(θ)),a)
    u2 = PlanarWave((;k,θ),a)

    x = CartesianCoordinates(1.0,2.3)

    @test u1(x) ≈ u2(x)
end

@testset "Single planar wave" begin

    k = 3.2
    θ = 0.3
    a = 3.4
    u = PlanarWave((;k,θ),a)

    @test u(CartesianCoordinates(0,0)) ≈ a

    pml = XAlignedRectangularPML(1.0, 2.0)

    ν = 1.7
    ζ = 4.8
    X = convert(CartesianCoordinates, PMLCoordinates(0.0, ζ), pml)
    U = u(X)
    pml_coords = PMLCoordinates(ν, ζ)
    ν_bar = ν/pml.δ
    _tx, J = tx_and_jacobian(u, pml, PMLCoordinates(ν, ζ))
    @test u(_tx) ≈ U*(1-ν_bar)
    ε = 1e-8
    @test (tx(u, pml, PMLCoordinates(ν+ε, ζ)).x - _tx.x)/ε ≈ J[:,1]
    @test (tx(u, pml, PMLCoordinates(ν, ζ+ε)).x - _tx.x)/ε ≈ J[:,2]

end

@testset "Sum of planar waves" begin

    PlanarWave(k,a=one(first(k))) = PlanarWave(SVector(k),a)
    PlanarWave((k,θ)::NamedTuple{(:k, :θ)},a=one(first(k))) = PlanarWave{2}(SVector(k*cos(θ), k*sin(θ)),a)

end
